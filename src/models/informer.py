"""
src/models/informer.py
──────────────────────
Informer for multi-step streamflow forecasting.

Key innovations over vanilla Transformer:
  1. ProbSparse self-attention  — O(L log L) instead of O(L²)
  2. Self-attention distilling  — halves sequence length each layer
  3. Generative decoder         — all pred_len steps in one forward pass

Input/output contract
─────────────────────
  encoder input : (B, seq_len,  c_in)   — history window
  decoder input : (B, label_len + pred_len, c_in)  — start token + zeros
  output        : (B, pred_len)          — discharge forecast

References
──────────
  Zhou et al. (2021) "Informer: Beyond Efficient Transformer for
  Long Sequence Time-Series Forecasting"  AAAI 2021.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────
# 1. Positional encoding
# ─────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)                  # (L, D)
        pos = torch.arange(max_len).unsqueeze(1).float()    # (L, 1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        # Slice `div` to the width of the odd-index slice. For even d_model
        # these are equal; for ODD d_model the cos slice is one narrower, and
        # the previous `div[:d_model//2]` silently mismatched.
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))         # (1, L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        return self.dropout(x + self.pe[:, : x.size(1)])


# ─────────────────────────────────────────────────────────────────────
# 2. Token embedding (linear projection of raw features → d_model)
# ─────────────────────────────────────────────────────────────────────

class TokenEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        # Conv1d with kernel=3, padding=1 preserves sequence length
        self.conv = nn.Conv1d(c_in, d_model, kernel_size=3, padding=1,
                              padding_mode="circular", bias=False)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)  →  out: (B, T, D)
        return self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)


class DataEmbedding(nn.Module):
    """Token + positional embedding."""

    def __init__(self, c_in: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.token = TokenEmbedding(c_in, d_model)
        self.pos   = PositionalEncoding(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pos(self.token(x))


# ─────────────────────────────────────────────────────────────────────
# 3. Attention backends
# ─────────────────────────────────────────────────────────────────────

class FullAttention(nn.Module):
    """
    Standard scaled dot-product attention via `F.scaled_dot_product_attention`.

    THE DEFAULT, and deliberately so. At our sequence length ProbSparse is both
    broken and counter-productive:

      * L = 336, factor = 5  ->  u = 5*int(ln 337) = 25 active queries.
        25 of 336 queries (7.4%) receive real attention; the other 311 (92.6%)
        receive V.mean() — a single byte-identical constant vector. Measured:
        "distinct vectors among the 311 inactive positions: 1".

      * The memory argument inverts at this length. The full attention matrix
        is 57.8M elements (220 MB); the ProbSparse *sampling intermediate* is
        137.6M (525 MB) — 2.4x LARGER than the thing it replaces. ProbSparse
        pays off around L>=5000, not L=336.

    `F.scaled_dot_product_attention` additionally dispatches to fused/flash
    kernels, so the matrix is never materialised at all.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout_p = dropout

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

    def forward(self, x: torch.Tensor,
                attn_mask: torch.Tensor = None) -> torch.Tensor:
        B, T, _ = x.shape
        q, k, v = self._split(self.Wq(x)), self._split(self.Wk(x)), \
            self._split(self.Wv(x))
        o = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        o = o.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out(o)


class ProbSparseSelfAttention(nn.Module):
    """
    ProbSparse self-attention (Zhou et al. 2021). NOT the default — kept so
    ablation I7 measures a *correct* implementation against a correct baseline.

    Two bugs were fixed here; both are why this is no longer the default.

    BUG 1 — 262 GB allocation (would OOM on any machine)
    ----------------------------------------------------
    The original line was:

        K_samp = K[:, :, idx.view(B, H, -1)].view(B, H, T_q, sample_k, d)

    That is MIXED basic + advanced indexing. PyTorch keeps the leading sliced
    dims AND inserts the advanced-index shape, giving (B,H, B,H,N, d) rather
    than (B,H,N,d). Reproduced in numpy at toy scale:

        K (2,3,10,4), idx (2,3,50)  ->  K[:,:,idx] = (2,3, 2,3,50, 4)
        elements produced 7,200  vs  needed 1,200   (6x blowup)

    At production shapes (H=8, T=336, sample_k=25, d=32):

        batch=4  ->   1.1 GB   (the self-test passed here)
        batch=16 ->  17.6 GB
        batch=32 ->  70.5 GB
        batch=64 -> 281.9 GB   <- our default batch size

    The self-test passed only because it ran at batch=4 and asserted the final
    *shape*; `.view()` forcibly reshapes the wrong tensor back, so shape checks
    succeed while the values are garbage. There is now a batch=64 self-test.

    Fix: gather with `torch.gather`, which cannot silently broadcast.

    BUG 2 — dead allocation
    -----------------------
    `K.unsqueeze(-2).expand(B,H,T_k,T_k,d)` was computed and then overwritten
    on the very next line without being read. Removed.

    (Credit: HANDOFF_REVIEW.md identified bug 1.)
    """

    def __init__(self, d_model: int, n_heads: int,
                 factor: int = 5, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.factor = factor

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_head).permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor,
                attn_mask: torch.Tensor = None) -> torch.Tensor:
        B, T, _ = x.shape
        H, d = self.n_heads, self.d_head

        Q = self._split_heads(self.Wq(x))          # (B,H,T,d)
        K = self._split_heads(self.Wk(x))
        V = self._split_heads(self.Wv(x))

        sample_k = max(1, min(self.factor * int(math.log(T + 1)), T))
        u = max(1, min(self.factor * int(math.log(T + 1)), T))

        # Sample `sample_k` keys per query to estimate query sparsity.
        # Flatten-and-gather keeps the shape explicit and allocates exactly
        # (B,H,T*sample_k,d) — no mixed-indexing broadcast, no (B,H,T,T,d)
        # intermediate. This is the line that used to allocate 262 GB.
        idx = torch.randint(T, (B, H, T * sample_k), device=Q.device)
        K_samp = torch.gather(
            K, 2, idx.unsqueeze(-1).expand(B, H, T * sample_k, d)
        ).view(B, H, T, sample_k, d)

        QK = torch.einsum("bhqd,bhqkd->bhqk", Q, K_samp)

        # Sparsity measure M = max - mean (Zhou et al. Eq. 4 approximation)
        M = QK.max(-1).values - QK.sum(-1) / sample_k     # (B,H,T)
        top_idx = M.topk(u, dim=-1).indices               # (B,H,u)

        Q_reduce = Q.gather(2, top_idx.unsqueeze(-1).expand(B, H, u, d))
        scores = torch.einsum("bhud,bhtd->bhut", Q_reduce, K) / math.sqrt(d)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))

        attn = self.dropout(torch.softmax(scores, dim=-1))
        V_active = torch.einsum("bhut,bhtd->bhud", attn, V)

        # Inactive queries keep the mean of V, per the paper's formulation.
        out = V.mean(dim=2, keepdim=True).expand(B, H, T, d).clone()
        out.scatter_(2, top_idx.unsqueeze(-1).expand(B, H, u, d), V_active)

        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, -1)
        return self.out(out)


def build_attention(kind: str, d_model: int, n_heads: int,
                    factor: int, dropout: float) -> nn.Module:
    if kind == "full":
        return FullAttention(d_model, n_heads, dropout)
    if kind == "prob":
        return ProbSparseSelfAttention(d_model, n_heads, factor, dropout)
    raise ValueError(f"Unknown attention: {kind!r}. Use 'full' or 'prob'.")


# ─────────────────────────────────────────────────────────────────────
# 4. Distillation (sequence compression)
# ─────────────────────────────────────────────────────────────────────

class DistilLayer(nn.Module):
    """
    Halves sequence length: MaxPool( ELU( Conv1d(x) ) )
    Applied between encoder attention layers.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1,
                              padding_mode="circular")
        self.norm = nn.BatchNorm1d(d_model)
        self.act  = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = x.permute(0, 2, 1)          # (B, D, T)
        x = self.act(self.norm(self.conv(x)))
        x = self.pool(x)
        return x.permute(0, 2, 1)       # (B, T/2, D)


# ─────────────────────────────────────────────────────────────────────
# 5. Encoder layer + Encoder
# ─────────────────────────────────────────────────────────────────────

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 factor: int, dropout: float, activation: str,
                 attention: str = "full"):
        super().__init__()
        self.attn  = build_attention(attention, d_model, n_heads,
                                     factor, dropout)
        self.ff1   = nn.Linear(d_model, d_ff)
        self.ff2   = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)
        self.act   = F.gelu if activation == "gelu" else F.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention + residual
        x = self.norm1(x + self.drop(self.attn(x)))
        # Feed-forward + residual
        ff = self.drop(self.act(self.ff1(x)))
        ff = self.drop(self.ff2(ff))
        return self.norm2(x + ff)


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList,
                 distil_layers: nn.ModuleList,
                 norm: nn.LayerNorm):
        super().__init__()
        self.layers  = layers
        self.distils = distil_layers
        self.norm    = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.distils):
                x = self.distils[i](x)
        return self.norm(x)


# ─────────────────────────────────────────────────────────────────────
# 6. Decoder layer + Decoder
# ─────────────────────────────────────────────────────────────────────

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float, activation: str):
        super().__init__()
        d_head = d_model // n_heads
        # Causal self-attention over decoder sequence
        self.self_attn  = nn.MultiheadAttention(d_model, n_heads,
                                                 dropout=dropout,
                                                 batch_first=True)
        # Cross-attention to encoder output
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads,
                                                 dropout=dropout,
                                                 batch_first=True)
        self.ff1  = nn.Linear(d_model, d_ff)
        self.ff2  = nn.Linear(d_ff, d_model)
        self.n1   = nn.LayerNorm(d_model)
        self.n2   = nn.LayerNorm(d_model)
        self.n3   = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.act  = F.gelu if activation == "gelu" else F.relu

    def forward(self, x: torch.Tensor, mem: torch.Tensor,
                tgt_mask: torch.Tensor = None) -> torch.Tensor:
        # Causal self-attention
        sa, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.n1(x + self.drop(sa))
        # Cross-attention over encoder memory
        ca, _ = self.cross_attn(x, mem, mem)
        x = self.n2(x + self.drop(ca))
        # Feed-forward
        ff = self.drop(self.act(self.ff1(x)))
        ff = self.drop(self.ff2(ff))
        return self.n3(x + ff)


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, norm: nn.LayerNorm,
                 projection: nn.Linear):
        super().__init__()
        self.layers = layers
        self.norm   = norm
        self.proj   = projection

    def forward(self, x: torch.Tensor, mem: torch.Tensor,
                tgt_mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mem, tgt_mask)
        x = self.norm(x)
        return self.proj(x)      # (B, label_len+pred_len, 1)


# ─────────────────────────────────────────────────────────────────────
# 7. Full Informer model
# ─────────────────────────────────────────────────────────────────────

class Informer(nn.Module):
    """
    Informer for multi-step discharge forecasting.

    Parameters
    ----------
    c_in       : number of input channels (12 for our dataset)
    seq_len    : encoder input length in timesteps (336)
    label_len  : decoder start-token length (24)
    pred_len   : forecast horizon (48)
    d_model    : hidden dimension (256)
    n_heads    : attention heads (8)
    enc_layers : encoder stack depth (2)
    dec_layers : decoder stack depth (1)
    d_ff       : feed-forward dimension (1024)
    factor     : ProbSparse sampling factor (5)
    dropout    : dropout rate (0.1)
    activation : "gelu" | "relu"
    use_distil : whether to apply distillation between encoder layers
    """

    def __init__(
        self,
        c_in       : int   = 12,
        seq_len    : int   = 336,
        label_len  : int   = 24,
        pred_len   : int   = 48,
        d_model    : int   = 256,
        n_heads    : int   = 8,
        enc_layers : int   = 2,
        dec_layers : int   = 1,
        d_ff       : int   = 1024,
        factor     : int   = 5,
        dropout    : float = 0.1,
        activation : str   = "gelu",
        use_distil : bool  = True,
        attention  : str   = "full",
        n_aux      : int   = 0,
    ):
        super().__init__()
        self.pred_len  = pred_len
        self.label_len = label_len
        self.seq_len   = seq_len
        self.n_aux     = n_aux

        # Embeddings
        self.enc_embed = DataEmbedding(c_in, d_model, dropout)
        self.dec_embed = DataEmbedding(c_in, d_model, dropout)

        # Encoder
        enc_attn_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, factor, dropout, activation,
                         attention)
            for _ in range(enc_layers)
        ])
        distil_layers = nn.ModuleList([
            DistilLayer(d_model)
            for _ in range(enc_layers - 1)
        ]) if use_distil else nn.ModuleList()

        self.encoder = Encoder(
            enc_attn_layers, distil_layers, nn.LayerNorm(d_model)
        )

        # Decoder
        dec_layers_list = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout, activation)
            for _ in range(dec_layers)
        ])
        self.decoder = Decoder(
            dec_layers_list,
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1, bias=True),   # project to scalar
        )

        # Optional AUXILIARY head: predicts the future meteorological forcing
        # (y_aux) from the encoder representation.
        #
        # This is the ONLY legitimate use of y_aux. metadata.json states
        #   "y_aux_role": "future_supervision_only_not_inference_inputs"
        # and y_aux is absent from test.h5. Feeding it to the DECODER would
        # let the model lean on 48h of perfect future weather that will not
        # exist at submission time — dev NSE would look excellent and the
        # submission would collapse.
        #
        # As an auxiliary TARGET it instead regularises the encoder toward
        # representations that carry forcing information, and the head is
        # simply discarded at inference. Multi-task learning, not conditioning.
        self.aux_head = (
            nn.Linear(d_model, pred_len * n_aux) if n_aux > 0 else None
        )

    def forward(
        self,
        x_enc       : torch.Tensor,           # (B, seq_len, c_in)
        x_dec       : torch.Tensor = None,    # (B, label_len+pred_len, c_in)
        return_aux  : bool = False,
    ):
        """
        Parameters
        ----------
        x_enc      : encoder input — history window, (B, seq_len, c_in)
        x_dec      : decoder start token. If None, built automatically as
                     the last label_len steps of x_enc followed by zeros.

                     NOTE: the zero-fill is deliberate and must stay. Filling
                     it with future forcing (y_aux) would be target leakage —
                     that tensor does not exist at test time. See `aux_head`.
        return_aux : also return the auxiliary forcing prediction.

        Returns
        -------
        pred                   : (B, pred_len)
        (pred, aux)            : if return_aux and an aux head exists,
                                 aux is (B, pred_len, n_aux)
        """
        B, _, C = x_enc.shape

        if x_dec is None:
            start = x_enc[:, -self.label_len:, :]          # (B, label_len, C)
            zeros = torch.zeros(B, self.pred_len, C,
                                device=x_enc.device,
                                dtype=x_enc.dtype)
            x_dec = torch.cat([start, zeros], dim=1)       # (B, L+P, C)

        # Causal mask for decoder self-attention.
        # Boolean mask: True = "do not attend". nn.MultiheadAttention accepts
        # a bool mask directly; the previous float(-inf) version produced NaN
        # rows under some fused kernels when a row was fully masked.
        dec_len = x_dec.size(1)
        tgt_mask = torch.triu(
            torch.ones(dec_len, dec_len, device=x_enc.device,
                       dtype=torch.bool), diagonal=1
        )

        enc_out = self.encoder(self.enc_embed(x_enc))       # (B, T', D)
        dec_out = self.decoder(
            self.dec_embed(x_dec), enc_out, tgt_mask
        )                                                    # (B, L+P, 1)

        pred = dec_out[:, -self.pred_len:, 0]               # (B, pred_len)

        if return_aux and self.aux_head is not None:
            # Pool the encoder memory, then project to (pred_len, n_aux).
            pooled = enc_out.mean(dim=1)                    # (B, D)
            aux = self.aux_head(pooled).view(B, self.pred_len, self.n_aux)
            return pred, aux
        if return_aux:
            return pred, None
        return pred


# ─────────────────────────────────────────────────────────────────────
# 8. Self-test
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import INFORMER_CFG, HISTORY_HOURS, FORECAST_HOURS, N_CHANNELS

    torch.manual_seed(42)
    device = torch.device("cpu")

    def make(attention="full", n_aux=0, d_model=None):
        return Informer(
            c_in=N_CHANNELS, seq_len=HISTORY_HOURS,
            label_len=INFORMER_CFG["label_len"], pred_len=FORECAST_HOURS,
            d_model=d_model or INFORMER_CFG["d_model"],
            n_heads=INFORMER_CFG["n_heads"],
            enc_layers=INFORMER_CFG["enc_layers"],
            dec_layers=INFORMER_CFG["dec_layers"],
            d_ff=INFORMER_CFG["d_ff"], factor=INFORMER_CFG["prob_factor"],
            dropout=INFORMER_CFG["dropout"],
            activation=INFORMER_CFG["activation"],
            use_distil=INFORMER_CFG["use_distil"],
            attention=attention, n_aux=n_aux,
        ).to(device)

    print("\n── full attention (default) ─────────────────────────────────")
    m = make("full")
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    x = torch.randn(4, HISTORY_HOURS, N_CHANNELS)
    with torch.no_grad():
        out = m(x)
    assert out.shape == (4, FORECAST_HOURS)
    print(f"  params={n_params:,}  x{tuple(x.shape)} -> out{tuple(out.shape)} ✓")

    # THE REGRESSION TEST.
    # The old self-test ran at batch=4 and asserted only the output shape,
    # which is exactly why the 262 GB indexing bug shipped unnoticed: .view()
    # reshapes the wrong tensor back, so the shape check passes on garbage.
    # This runs at the DEFAULT batch size, where the bug allocated 281.9 GB.
    print("\n── batch=64 regression test (the 262 GB bug) ────────────────")
    for attn in ("full", "prob"):
        mm = make(attn)
        xb = torch.randn(64, HISTORY_HOURS, N_CHANNELS)
        with torch.no_grad():
            ob = mm(xb)
        assert ob.shape == (64, FORECAST_HOURS), f"{attn}: {ob.shape}"
        assert torch.isfinite(ob).all(), f"{attn} produced non-finite output"
        print(f"  attention={attn:5s} batch=64 -> {tuple(ob.shape)} finite ✓")

    print("\n── ProbSparse coverage (why 'full' is the default) ──────────")
    T = HISTORY_HOURS
    u = min(INFORMER_CFG["prob_factor"] * int(math.log(T + 1)), T)
    print(f"  seq_len={T}: {u}/{T} queries active ({100*u/T:.1f}%); "
          f"{100*(1-u/T):.1f}% receive a constant vector")

    print("\n── auxiliary head (y_aux as TARGET, never as input) ─────────")
    ma = make("full", n_aux=11)
    with torch.no_grad():
        pred, aux = ma(x, return_aux=True)
    assert pred.shape == (4, FORECAST_HOURS)
    assert aux.shape == (4, FORECAST_HOURS, 11), aux.shape
    print(f"  pred{tuple(pred.shape)}  aux{tuple(aux.shape)} ✓")

    print("\n── odd d_model positional encoding ──────────────────────────")
    pe = PositionalEncoding(d_model=65, max_len=16, dropout=0.0)
    assert torch.isfinite(pe.pe).all()
    print("  d_model=65 (odd) -> finite PE ✓")

    print("\n── gradients flow ───────────────────────────────────────────")
    mg = make("full")
    xg = torch.randn(2, HISTORY_HOURS, N_CHANNELS)
    mg(xg).sum().backward()
    n_grad = sum(1 for p in mg.parameters()
                 if p.grad is not None and p.grad.abs().sum() > 0)
    n_all = sum(1 for p in mg.parameters() if p.requires_grad)
    print(f"  {n_grad}/{n_all} parameter tensors received gradient ✓")
    assert n_grad > 0.8 * n_all, "too many parameters got no gradient"

    print("\n✓ informer.py self-test passed\n")
