"""
src/models/dmel.py
──────────────────
DMEL model assembler.

Two operating modes (controlled by `use_ceemdan`):

  mode=False  [baseline — default]
    Single Informer processes the full input directly.
    No decomposition. Fast, no leakage risk.
    Used for ablation A-OUR-1 and as the starting baseline.

  mode=True   [full DMEL]
    Expects pre-computed RIMF components fed as x_high / x_low.
    Informer handles high-frequency RIMFs.
    LSTM handles low-frequency RIMFs.
    Outputs are summed (additive superposition).

Combiner variants (controlled by `combine`):
  "sum"     — fixed additive sum  (paper default)
  "linear"  — learned scalar weights per branch
  "mlp"     — small MLP over concatenated branch outputs

Basin embedding (controlled by `use_basin_emb`):
  If True, a learned embedding of dimension `basin_emb_dim` is
  appended to every time step before encoding.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    INFORMER_CFG, LSTM_CFG, ENSEMBLE_CFG, CEEMDAN_CFG, TARGET_CHANNEL,
    HISTORY_HOURS, FORECAST_HOURS, N_CHANNELS, N_AUX_CHANNELS, N_BASINS,
)
from models.informer   import Informer
from models.lstm_branch import LSTMBranch


# ─────────────────────────────────────────────────────────────────────
# Combiner variants
# ─────────────────────────────────────────────────────────────────────

class SumCombiner(nn.Module):
    """Fixed sum — no parameters."""
    def forward(self, preds):           # preds: list of (B, P)
        return sum(preds)


class LinearCombiner(nn.Module):
    """Learned scalar weight per branch, softmax-normalised."""
    def __init__(self, n_branches: int):
        super().__init__()
        self.w = nn.Parameter(torch.ones(n_branches))

    def forward(self, preds):
        w = torch.softmax(self.w, dim=0)
        return sum(w[i] * p for i, p in enumerate(preds))


class MLPCombiner(nn.Module):
    """Small MLP over concatenated branch outputs."""
    def __init__(self, n_branches: int, pred_len: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_branches * pred_len, pred_len * 2),
            nn.GELU(),
            nn.Linear(pred_len * 2, pred_len),
        )

    def forward(self, preds):           # preds: list of (B, P)
        return self.net(torch.cat(preds, dim=-1))


def build_combiner(method: str, n_branches: int, pred_len: int) -> nn.Module:
    if method == "sum":
        return SumCombiner()
    elif method == "linear":
        return LinearCombiner(n_branches)
    elif method == "mlp":
        return MLPCombiner(n_branches, pred_len)
    else:
        raise ValueError(f"Unknown combiner: {method!r}. "
                         "Choose 'sum', 'linear', or 'mlp'.")


# ─────────────────────────────────────────────────────────────────────
# DMEL model
# ─────────────────────────────────────────────────────────────────────

class DMEL(nn.Module):
    """
    Dual-Modal Ensemble Learning model.

    Parameters
    ----------
    c_in          : input channels (12)
    seq_len       : encoder sequence length (336)
    label_len     : Informer decoder start-token length (24)
    pred_len      : forecast steps (48)
    use_ceemdan   : if True, expects pre-decomposed RIMF tensors
    n_high        : number of high-freq RIMF branches (used when use_ceemdan=True)
    n_low         : number of low-freq  RIMF branches (used when use_ceemdan=True)
    shared_informer : if True, share one Informer across all HF branches
    shared_lstm     : if True, share one LSTM across all LF branches
    combine       : "sum" | "linear" | "mlp"
    use_basin_emb : append learned basin embedding to input
    basin_emb_dim : dimension of basin embedding
    informer_cfg  : dict of Informer kwargs (from config.py)
    lstm_cfg      : dict of LSTMBranch kwargs (from config.py)
    residual_output : baseline mode only. Predict the CHANGE from the last
                    observed discharge: pred = f(x) + x[:, -1, residual_idx].
                    A zero network then forecasts persistence exactly, which
                    targets the measured h+1..3 / steady-flow deficit.
    residual_idx  : column of discharge in x (TARGET_CHANNEL unless
                    input_channels subsets it)
    base_model    : baseline mode only. "informer" | "lstm"
    """

    def __init__(
        self,
        c_in            : int   = N_CHANNELS,
        seq_len         : int   = HISTORY_HOURS,
        label_len       : int   = INFORMER_CFG["label_len"],
        pred_len        : int   = FORECAST_HOURS,
        use_ceemdan     : bool  = False,
        n_high          : int   = CEEMDAN_CFG["n_high"],
        n_low           : int   = CEEMDAN_CFG["n_low"],
        shared_informer : bool  = False,
        shared_lstm     : bool  = False,
        combine         : str   = ENSEMBLE_CFG["method"],
        use_basin_emb   : bool  = False,
        basin_emb_dim   : int   = 16,
        informer_cfg    : dict  = None,
        lstm_cfg        : dict  = None,
        n_aux           : int   = 0,
        residual_output : bool  = False,
        residual_idx    : int   = TARGET_CHANNEL,
        base_model      : str   = "informer",
    ):
        super().__init__()
        self.use_ceemdan     = use_ceemdan
        self.pred_len        = pred_len
        self.use_basin_emb   = use_basin_emb
        self.n_high          = n_high
        self.n_low           = n_low
        self.shared_informer = shared_informer
        self.shared_lstm     = shared_lstm
        self.residual_output = residual_output
        self.residual_idx    = residual_idx
        self.base_model      = base_model

        if residual_output and use_ceemdan:
            # Each branch would add the last value again: the sum of K
            # residual branches predicts K x persistence.
            raise ValueError("residual_output applies to baseline mode only "
                             "(use_ceemdan=False)")

        icfg = informer_cfg or INFORMER_CFG
        lcfg = lstm_cfg     or LSTM_CFG

        # Optional basin embedding: appended to every timestep
        if use_basin_emb:
            self.basin_emb = nn.Embedding(N_BASINS, basin_emb_dim)
            eff_c_in = c_in + basin_emb_dim
        else:
            self.basin_emb = None
            eff_c_in = c_in

        def _make_informer(c):
            return Informer(
                c_in       = c,
                seq_len    = seq_len,
                label_len  = label_len,
                pred_len   = pred_len,
                d_model    = icfg["d_model"],
                n_heads    = icfg["n_heads"],
                enc_layers = icfg["enc_layers"],
                dec_layers = icfg["dec_layers"],
                d_ff       = icfg["d_ff"],
                factor     = icfg["prob_factor"],
                dropout    = icfg["dropout"],
                activation = icfg["activation"],
                use_distil = icfg["use_distil"],
                attention  = icfg.get("attention", "full"),
                n_aux      = n_aux,
                pad_mode   = icfg.get("pad_mode", "circular"),
            )

        def _make_lstm(c):
            return LSTMBranch(
                c_in          = c,
                hidden_size   = lcfg["hidden_size"],
                num_layers    = lcfg["num_layers"],
                pred_len      = pred_len,
                dropout       = lcfg["dropout"],
                bidirectional = lcfg["bidirectional"],
            )

        # ── Baseline mode: single Informer, full input ────────────────
        # NOTE: this used to `return` before building the combiner, which made
        # `combine` unreachable in every default run and turned exp_010/011/
        # 011b (the three combiner ablations) into silent no-ops. With one
        # branch there is nothing to combine, so a combiner is genuinely not
        # needed here — but the flag is now validated instead of ignored.
        if not use_ceemdan:
            if base_model == "informer":
                self.informer = _make_informer(eff_c_in)
            elif base_model == "lstm":
                if n_aux:
                    raise ValueError("aux_task needs the Informer encoder; "
                                     "base_model='lstm' has no aux head")
                self.lstm_base = _make_lstm(eff_c_in)
            else:
                raise ValueError(f"Unknown base_model: {base_model!r}")
            self.combiner = None
            if combine != "sum":
                raise ValueError(
                    f"ensemble_method={combine!r} has no effect when "
                    "use_ceemdan=False (there is only one branch). "
                    "Set use_ceemdan=True to make this ablation meaningful.")
            return

        # ── DMEL mode: dual-channel with RIMF routing ─────────────────
        # High-frequency branches (Informer)
        if shared_informer:
            self.hf_informer = _make_informer(eff_c_in)
            self.hf_branches = None
        else:
            self.hf_branches = nn.ModuleList(
                [_make_informer(eff_c_in) for _ in range(n_high)]
            )
            self.hf_informer = None

        # Low-frequency branches (LSTM)
        if shared_lstm:
            self.lf_lstm    = _make_lstm(eff_c_in)
            self.lf_branches = None
        else:
            self.lf_branches = nn.ModuleList(
                [_make_lstm(eff_c_in) for _ in range(n_low)]
            )
            self.lf_lstm = None

        total_branches = n_high + n_low
        self.combiner  = build_combiner(combine, total_branches, pred_len)

    # ── helpers ───────────────────────────────────────────────────────

    def _add_basin_emb(self, x: torch.Tensor,
                        basin_ids: torch.Tensor) -> torch.Tensor:
        """Append basin embedding to every timestep of x."""
        emb = self.basin_emb(basin_ids)                 # (B, emb_dim)
        emb = emb.unsqueeze(1).expand(-1, x.size(1), -1)  # (B, T, emb_dim)
        return torch.cat([x, emb], dim=-1)              # (B, T, C+emb)

    # ── forward ───────────────────────────────────────────────────────

    def forward(
        self,
        x          : torch.Tensor,             # (B, T, C)
        basin_ids  : torch.Tensor = None,      # (B,)  required if use_basin_emb
        x_high     : list         = None,      # list of (B,T,C) high-freq RIMFs
        x_low      : list         = None,      # list of (B,T,C) low-freq  RIMFs
        x_dec      : torch.Tensor = None,      # (B, label+pred, C) optional
        return_aux : bool         = False,
    ):
        """
        Returns
        -------
        pred : (B, pred_len)
        """
        # Optionally inject basin embedding
        if self.use_basin_emb and basin_ids is not None:
            x = self._add_basin_emb(x, basin_ids)

        # ── Baseline: single Informer on full input ───────────────────
        if not self.use_ceemdan:
            if self.base_model == "lstm":
                out = self.lstm_base(x)
                out = (out, None) if return_aux else out
            else:
                out = self.informer(x, x_dec, return_aux=return_aux)
            if not self.residual_output:
                return out
            # x[:, -1, residual_idx] is the last observed discharge in the
            # same per-basin z-units as y (dataset.py normalises both with the
            # target channel's stats).
            last = x[:, -1, self.residual_idx].unsqueeze(1)    # (B, 1)
            if return_aux:
                return out[0] + last, out[1]
            return out + last

        # ── DMEL: dual-channel ────────────────────────────────────────
        if x_high is None or x_low is None:
            raise ValueError(
                "use_ceemdan=True requires x_high and x_low RIMF lists."
            )

        # Add basin embedding to each component if needed
        if self.use_basin_emb and basin_ids is not None:
            x_high = [self._add_basin_emb(xh, basin_ids) for xh in x_high]
            x_low  = [self._add_basin_emb(xl, basin_ids) for xl in x_low]

        # A branch-count mismatch used to allocate dead Informers silently:
        # n_high defaulted to 5 here while the router sent 2 tensors, so 56%
        # of the reported parameters never received input or gradient.
        if not self.shared_informer and len(x_high) != len(self.hf_branches):
            raise ValueError(f"{len(x_high)} HF inputs for "
                             f"{len(self.hf_branches)} HF branches")
        if not self.shared_lstm and len(x_low) != len(self.lf_branches):
            raise ValueError(f"{len(x_low)} LF inputs for "
                             f"{len(self.lf_branches)} LF branches")

        # High-frequency predictions
        hf_preds = []
        aux_out = None
        for i, xh in enumerate(x_high):
            branch = (self.hf_informer if self.shared_informer
                      else self.hf_branches[i])
            if return_aux and i == 0:
                p, aux_out = branch(xh, x_dec, return_aux=True)
                hf_preds.append(p)
            else:
                hf_preds.append(branch(xh, x_dec))

        # Low-frequency predictions
        lf_preds = []
        for i, xl in enumerate(x_low):
            branch = (self.lf_lstm if self.shared_lstm
                      else self.lf_branches[i])
            lf_preds.append(branch(xl))

        all_preds = hf_preds + lf_preds
        out = self.combiner(all_preds)
        return (out, aux_out) if return_aux else out


# ─────────────────────────────────────────────────────────────────────
# Factory function — builds model from a flat config dict
# ─────────────────────────────────────────────────────────────────────

def build_model(cfg: dict) -> DMEL:
    """
    Build a DMEL model from a flat experiment config dict.

    The cfg dict is the merged TRAIN_CFG + experiment overrides.
    This is the single entry point used by experiment.py.

    c_in is DERIVED from input_channels when that is set, not read blindly
    from cfg. The dataset subsets X to the requested channels, so a config
    carrying input_channels=[11] with c_in=12 builds a conv expecting 12
    channels and is handed 1 — an immediate RuntimeError ~90 s into training:

        weight of size [256, 12, 3], expected input[64, 1, 338] to have
        12 channels, but got 1 channels instead

    That is what killed exp_003__discharge_only and exp_004__precip_discharge.
    An explicit cfg["c_in"] still wins, so a deliberate override is possible.
    """
    input_channels = cfg.get("input_channels")
    default_c_in = (len(input_channels) if input_channels is not None
                    else N_CHANNELS)
    # rimf_append: the K cached RIMFs ride along as extra input channels.
    if cfg.get("rimf_append", False):
        default_c_in += cfg.get("n_rimf", CEEMDAN_CFG["n_rimf"])

    # Discharge column after channel subsetting; appended RIMFs and a basin
    # embedding go AFTER the raw channels, so the index is unaffected.
    residual_idx = TARGET_CHANNEL
    if cfg.get("residual_output", False) and input_channels is not None:
        if TARGET_CHANNEL not in input_channels:
            raise ValueError("residual_output needs the discharge channel "
                             f"({TARGET_CHANNEL}) in input_channels")
        residual_idx = list(input_channels).index(TARGET_CHANNEL)

    return DMEL(
        c_in            = cfg.get("c_in") or default_c_in,
        seq_len         = cfg.get("seq_len",          HISTORY_HOURS),
        label_len       = cfg.get("label_len",        INFORMER_CFG["label_len"]),
        pred_len        = cfg.get("pred_len",         FORECAST_HOURS),
        use_ceemdan     = cfg.get("use_ceemdan",      False),
        # Defaults come from CEEMDAN_CFG — the same source the RIMF router
        # (decompose.assemble_branch_inputs) reads.
        n_high          = cfg.get("n_high",           CEEMDAN_CFG["n_high"]),
        n_low           = cfg.get("n_low",            CEEMDAN_CFG["n_low"]),
        shared_informer = cfg.get("shared_informer",  False),
        shared_lstm     = cfg.get("shared_lstm",      False),
        combine         = cfg.get("ensemble_method",  ENSEMBLE_CFG["method"]),
        use_basin_emb   = cfg.get("use_basin_emb",    False),
        basin_emb_dim   = cfg.get("basin_emb_dim",    16),
        n_aux           = (N_AUX_CHANNELS if cfg.get("aux_task", False) else 0),
        residual_output = cfg.get("residual_output",  False),
        residual_idx    = residual_idx,
        base_model      = cfg.get("base_model",       "informer"),
        informer_cfg    = {
            "d_model"   : cfg.get("d_model",    INFORMER_CFG["d_model"]),
            "n_heads"   : cfg.get("n_heads",    INFORMER_CFG["n_heads"]),
            "enc_layers": cfg.get("enc_layers", INFORMER_CFG["enc_layers"]),
            "dec_layers": cfg.get("dec_layers", INFORMER_CFG["dec_layers"]),
            "d_ff"      : cfg.get("d_ff",       INFORMER_CFG["d_ff"]),
            "prob_factor": cfg.get("prob_factor", INFORMER_CFG["prob_factor"]),
            "dropout"   : cfg.get("dropout",    INFORMER_CFG["dropout"]),
            "activation": cfg.get("activation", INFORMER_CFG["activation"]),
            "use_distil": cfg.get("use_distil", INFORMER_CFG["use_distil"]),
            "attention" : cfg.get("attention",  INFORMER_CFG["attention"]),
            "label_len" : cfg.get("label_len",  INFORMER_CFG["label_len"]),
            "pad_mode"  : cfg.get("pad_mode",   INFORMER_CFG["pad_mode"]),
        },
        lstm_cfg        = {
            "hidden_size"  : cfg.get("lstm_hidden",   LSTM_CFG["hidden_size"]),
            "num_layers"   : cfg.get("lstm_layers",   LSTM_CFG["num_layers"]),
            "dropout"      : cfg.get("lstm_dropout",  LSTM_CFG["dropout"]),
            "bidirectional": cfg.get("lstm_bidir",    LSTM_CFG["bidirectional"]),
        },
    )


# ─────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    B, T, C, P = 4, HISTORY_HOURS, N_CHANNELS, FORECAST_HOURS

    # ── Test 1: baseline mode (no CEEMDAN) ────────────────────────────
    print("\n── DMEL baseline (use_ceemdan=False) ────────────────────────")
    m = DMEL(c_in=C, seq_len=T, pred_len=P, use_ceemdan=False)
    x = torch.randn(B, T, C)
    with torch.no_grad():
        out = m(x)
    assert out.shape == (B, P), f"Got {out.shape}"
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"  params={n:,}  output{tuple(out.shape)}  ✓")

    # ── Test 2: baseline + basin embedding ────────────────────────────
    print("\n── DMEL baseline + basin_emb ────────────────────────────────")
    m2 = DMEL(c_in=C, seq_len=T, pred_len=P, use_ceemdan=False,
              use_basin_emb=True, basin_emb_dim=16)
    bids = torch.randint(0, N_BASINS, (B,))
    with torch.no_grad():
        out2 = m2(x, basin_ids=bids)
    assert out2.shape == (B, P)
    print(f"  output{tuple(out2.shape)}  ✓")

    # ── Test 3: DMEL dual-channel mode ────────────────────────────────
    print("\n── DMEL dual-channel (use_ceemdan=True, n_high=2, n_low=1) ──")
    m3 = DMEL(c_in=C, seq_len=T, pred_len=P, use_ceemdan=True,
              n_high=2, n_low=1, combine="sum")
    x_high = [torch.randn(B, T, C) for _ in range(2)]
    x_low  = [torch.randn(B, T, C) for _ in range(1)]
    with torch.no_grad():
        out3 = m3(x, x_high=x_high, x_low=x_low)
    assert out3.shape == (B, P)
    print(f"  output{tuple(out3.shape)}  ✓")

    # ── Test 4: MLP combiner ──────────────────────────────────────────
    print("\n── DMEL dual-channel + MLP combiner ─────────────────────────")
    m4 = DMEL(c_in=C, seq_len=T, pred_len=P, use_ceemdan=True,
              n_high=2, n_low=1, combine="mlp")
    with torch.no_grad():
        out4 = m4(x, x_high=x_high, x_low=x_low)
    assert out4.shape == (B, P)
    print(f"  output{tuple(out4.shape)}  ✓")

    # ── Test 5: build_model factory ───────────────────────────────────
    print("\n── build_model() factory ────────────────────────────────────")
    cfg = {"use_ceemdan": False, "use_basin_emb": False}
    m5  = build_model(cfg)
    with torch.no_grad():
        out5 = m5(x)
    assert out5.shape == (B, P)
    print(f"  output{tuple(out5.shape)}  ✓")

    print("\n✓ dmel.py self-test passed\n")
