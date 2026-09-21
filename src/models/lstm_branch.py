"""
src/models/lstm_branch.py
─────────────────────────
LSTM branch for low-frequency discharge forecasting.

Takes the full input window (B, T, C) and outputs a pred_len forecast.

Architecture
────────────
  Input projection  : Linear(c_in → hidden_size)
  LSTM stack        : num_layers of LSTM cells with optional dropout
  Output projection : Linear(hidden_size → pred_len)

The branch can also accept a single-channel input (one RIMF component
after CEEMDAN decomposition) by setting c_in=1.
"""

import torch
import torch.nn as nn


class LSTMBranch(nn.Module):
    """
    LSTM branch for multi-step forecasting.

    Parameters
    ----------
    c_in        : number of input channels
    hidden_size : LSTM hidden state size (256)
    num_layers  : number of stacked LSTM layers (2)
    pred_len    : number of forecast steps to output (48)
    dropout     : dropout between LSTM layers (0.1)
    bidirectional : use bidirectional LSTM (False by default)
    """

    def __init__(
        self,
        c_in         : int   = 12,
        hidden_size  : int   = 256,
        num_layers   : int   = 2,
        pred_len     : int   = 48,
        dropout      : float = 0.1,
        bidirectional: bool  = False,
    ):
        super().__init__()
        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.pred_len      = pred_len
        self.bidirectional = bidirectional
        self.n_dir         = 2 if bidirectional else 1

        # Project raw features to hidden dimension
        self.input_proj = nn.Linear(c_in, hidden_size)

        # LSTM stack
        # dropout only applied between layers (not on the last layer)
        self.lstm = nn.LSTM(
            input_size    = hidden_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            dropout       = dropout if num_layers > 1 else 0.0,
            bidirectional = bidirectional,
        )

        # Project final hidden state → all pred_len steps at once
        self.output_proj = nn.Linear(hidden_size * self.n_dir, pred_len)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Initialise forget gate bias to 1 (standard LSTM trick)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, c_in)  — input window (full history or one RIMF)

        Returns
        -------
        pred : (B, pred_len)  — forecast
        """
        # Project input features
        h = self.input_proj(x)                         # (B, T, hidden)

        # LSTM forward
        _, (h_n, _) = self.lstm(h)
        # h_n: (num_layers * n_dir, B, hidden)

        # Take last layer's hidden state
        if self.bidirectional:
            # Concatenate forward and backward last states
            h_last = torch.cat(
                [h_n[-2], h_n[-1]], dim=-1             # (B, 2*hidden)
            )
        else:
            h_last = h_n[-1]                           # (B, hidden)

        return self.output_proj(h_last)                # (B, pred_len)


# ─────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import LSTM_CFG, HISTORY_HOURS, FORECAST_HOURS, N_CHANNELS

    torch.manual_seed(42)

    model = LSTMBranch(
        c_in          = N_CHANNELS,
        hidden_size   = LSTM_CFG["hidden_size"],
        num_layers    = LSTM_CFG["num_layers"],
        pred_len      = FORECAST_HOURS,
        dropout       = LSTM_CFG["dropout"],
        bidirectional = LSTM_CFG["bidirectional"],
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  LSTMBranch  params={n_params:,}")

    B = 4
    x = torch.randn(B, HISTORY_HOURS, N_CHANNELS)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (B, FORECAST_HOURS), \
        f"Expected ({B},{FORECAST_HOURS}), got {out.shape}"
    print(f"  Forward pass:  x{tuple(x.shape)} → out{tuple(out.shape)}  ✓")
    print(f"  Output range:  [{out.min():.3f}, {out.max():.3f}]")
    print("\n✓ lstm_branch.py self-test passed\n")
