"""PC-FRA H16 residual adapters (frozen PSRC + frozen Chronos-2 prior).

Registered architecture (docs/pcfra_h16_experiment_protocol.md):

* Arm B (PARA-style): one window-level MLP that directly regresses the
  full H-dimensional residual vector::

      u    = concat(y_psrc, C_bar, D, L, E_pv, sigma, horizon_embedding)
      h    = Dropout(GELU(LayerNorm(Linear(u, 96))))
      raw  = Linear(h, H)
      y    = y_psrc + epsilon * tanh(raw)

* Arm C (PC-FRA): identical trunk and parameter budget, plus a single
  FiLM conditioning of the hidden state by the mean-pooled six physical
  semantic tokens already produced by the frozen PSRC::

      s = mean_pool(six_physical_tokens)
      gamma, beta = Linear(GELU(Linear(s, 192)), 192).chunk(2)
      h_tilde = (1 + 0.1 * tanh(gamma)) * h + 0.1 * tanh(beta)

The output layer is zero-initialised in both arms, so the very first
forward is bitwise identical to frozen PSRC (``epsilon * tanh(0) == 0``)
while residual gradients still flow once training starts. There is no
FM token, cross-attention, arbitration, confidence gate, or end-to-end
PSRC update; Chronos never enters the backbone or the physical-token
attention.
"""

from __future__ import annotations

import torch
from torch import nn

# --- Pack layout -----------------------------------------------------------
# Per-window feature vector carried alongside the batch (physical units for
# power quantities; D is precomputed in the globally-standardised target
# space because it references the partner window's frozen PSRC output).
#
#   [0, H)      C_bar             clipped/masked Chronos q0.5, physical
#   [H, 2H)     C_bar_partner     stratified-shuffle partner C_bar
#   [2H, 3H)    D_partner_std     partner disagreement, standardized
#   3H + 0      L_phys            latest observed PV (physical)
#   3H + 1      sigma_phys        recent PV volatility (physical)
#   3H + 2: +4  operating state   one-hot (night/low, ramp, peak, regular)
#   3H + 6      recent level      level / capacity (training z-scored later)
#   3H + 7: +4  intra-day bucket  one-hot (four fixed 6 h buckets)
STATE_DIM = 4
BUCKET_DIM = 4
EPV_DIM = STATE_DIM + 1 + BUCKET_DIM  # 9
SCALAR_DIM = 2  # L, sigma


def pack_width(horizon: int) -> int:
    return 3 * int(horizon) + SCALAR_DIM + EPV_DIM


def pack_slices(horizon: int) -> dict[str, slice | int]:
    h = int(horizon)
    return {
        "c_bar": slice(0, h),
        "c_bar_partner": slice(h, 2 * h),
        "d_partner": slice(2 * h, 3 * h),
        "latest": 3 * h,
        "sigma": 3 * h + 1,
        "state": slice(3 * h + 2, 3 * h + 2 + STATE_DIM),
        "level": 3 * h + 2 + STATE_DIM,
        "bucket": slice(3 * h + 3 + STATE_DIM, 3 * h + 3 + STATE_DIM + BUCKET_DIM),
    }


class PcFraResidualAdapter(nn.Module):
    """Window-level H-dim residual adapter (PARA trunk + optional FiLM)."""

    def __init__(
        self,
        horizon: int,
        hidden: int = 96,
        dropout: float = 0.1,
        use_film: bool = False,
        token_dim: int = 32,
        film_hidden: int = 192,
        film_bound: float = 0.1,
        horizon_embedding_dim: int = 8,
        epsilon: float = 1.0,
    ) -> None:
        super().__init__()
        horizon = int(horizon)
        hidden = int(hidden)
        if horizon <= 0 or hidden <= 0:
            raise ValueError("horizon and hidden must be positive")
        if not float(epsilon) > 0.0:
            raise ValueError("epsilon must be positive")
        self.horizon = horizon
        self.hidden = hidden
        self.use_film = bool(use_film)
        self.film_bound = float(film_bound)
        self.epsilon = float(epsilon)
        self.horizon_embedding_dim = int(horizon_embedding_dim)

        self.horizon_embedding = nn.Parameter(
            torch.empty(1, horizon, self.horizon_embedding_dim)
        )
        nn.init.normal_(self.horizon_embedding, std=0.02)

        # Four H-wide trajectories + E_pv (9) + sigma (1) + flat horizon ids.
        in_dim = 4 * horizon + EPV_DIM + 1 + horizon * self.horizon_embedding_dim
        self.input_linear = nn.Linear(in_dim, hidden)
        self.input_norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(float(dropout))
        if self.use_film:
            self.film = nn.Sequential(
                nn.Linear(int(token_dim), int(film_hidden)),
                nn.GELU(),
                nn.Linear(int(film_hidden), 2 * hidden),
            )
        self.output_linear = nn.Linear(hidden, horizon)
        # Identity at construction and whenever weights are re-zeroed:
        # epsilon * tanh(0) = 0 -> y == y_psrc bitwise.
        nn.init.zeros_(self.output_linear.weight)
        nn.init.zeros_(self.output_linear.bias)

    def forward(
        self,
        psrc: torch.Tensor,
        prior: torch.Tensor,
        disagreement: torch.Tensor,
        latest: torch.Tensor,
        epv: torch.Tensor,
        sigma: torch.Tensor,
        semantic_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        batch = psrc.shape[0]
        h = self.horizon
        if psrc.shape != (batch, h):
            raise ValueError(f"psrc must be [batch, {h}], got {tuple(psrc.shape)}")
        if prior.shape != psrc.shape or disagreement.shape != psrc.shape:
            raise ValueError("prior and disagreement must match psrc [batch, H]")
        if latest.shape != (batch, 1) or sigma.shape != (batch, 1):
            raise ValueError("latest and sigma must be [batch, 1]")
        if epv.shape != (batch, EPV_DIM):
            raise ValueError(f"epv must be [batch, {EPV_DIM}], got {tuple(epv.shape)}")
        features = torch.cat(
            [
                psrc,
                prior,
                disagreement,
                latest.expand(batch, h),
                epv,
                sigma,
                self.horizon_embedding.expand(batch, -1, -1).reshape(batch, -1),
            ],
            dim=-1,
        )
        hidden = self.dropout(
            torch.nn.functional.gelu(self.input_norm(self.input_linear(features)))
        )
        if self.use_film:
            if semantic_tokens is None:
                raise ValueError("PC-FRA FiLM requires the six physical tokens")
            if semantic_tokens.ndim != 3 or semantic_tokens.shape[0] != batch:
                raise ValueError("semantic_tokens must be [batch, K, token_dim]")
            summary = semantic_tokens.mean(dim=1)
            gamma, beta = self.film(summary).chunk(2, dim=-1)
            hidden = (
                1.0 + self.film_bound * torch.tanh(gamma)
            ) * hidden + self.film_bound * torch.tanh(beta)
        delta_raw = self.output_linear(hidden)
        return self.epsilon * torch.tanh(delta_raw)
