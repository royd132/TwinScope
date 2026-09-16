# PC-FRA H16 Experiment Protocol

Status: designed, not executed.

This protocol uses the two supplied PDF summaries as method references. They
do not override the repository's data split, checkpoint, or interface
contracts.

## Research Questions

1. Does a frozen Chronos-2 forecast prior add incremental value after the
   frozen PSRC forecast?
2. Does conditioning that residual adapter on the existing six physical
   states improve over the PARA-style adapter with the same prior and base
   forecast?

The numerical PSRC remains the forecaster. Chronos-2 is an offline, frozen
auxiliary prior.

## Fixed Setting

| Item | Registered value |
|---|---|
| Resolution | 15 min |
| Lookback / horizon | L96 -> H16 (24 h -> 4 h) |
| Split | chronological 70% / 15% / 15% |
| Base checkpoint | existing formal PSRC checkpoint, unchanged |
| Chronos model | local `amazon/chronos-2`; record package/model revision in cache metadata |
| Chronos input | target PV history plus only historical covariates available at issue time; future-known geometry may be supplied separately |
| Prior used by adapter | clipped native q0.5 trajectory |
| Adapter hidden width | 96 |
| Adapter dropout | 0.1 |
| Adapter optimizer | Adam, lr `1e-3`, weight decay `0` |
| Epochs / patience | 50 / 10 |
| Seed | 2026 for the registered reproduction run |
| Trainable parameters | adapter only; PSRC and Chronos stay frozen |
| Output | `y = y_psrc + epsilon * tanh(delta_raw)` |

`epsilon` is computed once from training windows only as the 90th percentile
of `abs(y - y_psrc)` in standardized target units, capped at `0.2 * capacity`
after conversion to physical units. It is then fixed for validation and test.
There is no learned global gate or learned lambda.

## Input Contract

For every legal window, construct the following from information available at
the forecast origin:

```text
C       = Chronos-2 q0.5 forecast
C_bar   = clip(C, 0, capacity) * daylight_mask
D       = C_bar - y_psrc
L       = latest observed PV, repeated over H
E_pv    = [one-hot operating state, normalized recent level,
           intra-day bucket]
sigma   = recent PV volatility
```

The operating state is registered before training and is not tuned:

```text
night/low: recent level < 0.20 capacity
peak:      recent level > 0.80 capacity
ramp:      max adjacent change in the last hour > 0.10 capacity
regular:   otherwise
```

Rules are applied in the order `night/low`, `ramp`, `peak`, `regular`.
Intra-day is four fixed 6-hour buckets. All continuous values are normalized
with training-only statistics. Future target, future weather observations,
and future quality flags are never passed to Chronos or the adapter.

## Main Arms

### A: Frozen PSRC control

The existing formal PSRC checkpoint is evaluated unchanged. It is not
retrained as part of this campaign.

### B: PARA-style Chronos residual adapter

Use exactly the same `C_bar`, `D`, `L`, `E_pv`, and `sigma` definitions above.
The adapter is:

```text
u = concat(y_psrc, C_bar, D, L, E_pv, sigma, horizon_embedding)
h = Dropout(GELU(LayerNorm(Linear(u, 96))))
delta_raw = Linear(h, H)
```

The last linear layer is zero-initialized, so the first forward pass is
bitwise identical to PSRC. There is no FM token, cross-attention, arbitration,
confidence gate, or end-to-end PSRC update.

### C: PC-FRA (ours)

B uses the same backbone inputs and parameter budget. The only registered
addition is the six physical semantic tokens already produced by PSRC:

```text
s = mean_pool(six_physical_tokens)
gamma, beta = Linear(GELU(Linear(s, 192)), 192).chunk(2)
h_tilde = (1 + 0.1 * tanh(gamma)) * h + 0.1 * tanh(beta)
delta_raw = Linear(h_tilde, H)
```

The `0.1` FiLM bound is fixed, not tuned. Physical tokens condition the
adapter hidden state; they do not become Chronos tokens and do not enter GTR,
L-Drive, Spectral MKAN, Multi-scale Patch, CorPatch, or PSRC attention.

## Negative Diagnostics

These are supplementary diagnostics, not extra SOTA rows and not used to tune
the main result:

- `C-phys-shuffle`: permute physical-token windows within the same month and
  intra-day bucket, preserving marginals but breaking physical alignment.
- `C-prior-shuffle`: permute `C_bar` and `D` with the same restriction,
  preserving seasonal/time-of-day marginals but breaking Chronos content.

The shuffling map is generated once with seed 2026 and stored beside the
protocol. If either diagnostic performs like C, the corresponding branch is
content-independent and the claimed mechanism is rejected.

## Leakage-Safe Execution Plan

### Phase 0: Interface audit

- Verify cache metadata: model revision, quantile order, dataset hash,
  context rows, and `target_pv_history_only` or the explicitly registered
  covariate schema.
- Verify physical clipping and standardized-to-physical conversion on three
  hand-checked windows.
- Verify B and C have identical PSRC and Chronos tensors before their adapter
  conditioning difference.
- Verify zero-initialized adapter output is bitwise equal to A.

### Phase 1: Development only

Use `dkasc_site31` and `pvod_station02` only. Train B and C on the training
split and select the best epoch from validation RMSE. Do not read, dump, or
rank test metrics. Existing `outputs/lcra_h16_20260915` test files are marked
exploratory and cannot be used as confirmatory evidence for this protocol.

Promotion rules are fixed:

- B must improve validation RMSE over A on both development stations.
- C must improve validation RMSE over B on both development stations.
- Neither promotion is allowed if MAE worsens on both stations.

If a rule fails, stop that line. Do not retune width, learning rate, clipping,
tokens, or attention after seeing the failure.

### Phase 2: Confirmatory run

After Phase 1 is frozen, evaluate A/B/C once on `dkasc_site9a` and
`pvod_station00` as untouched adapter-confirmation stations. Generate their
Chronos caches before the adapter checkpoint is selected, but do not inspect
their test metrics. Then evaluate the frozen selected checkpoints on all four
stations once for the descriptive four-station table; label Site31 and
PVOD02 as development results because their adapter test was previously
inspected.

No new architecture or hyperparameter decision is permitted in Phase 2.

## Reporting

Primary metrics: physical-scale RMSE and MAE. Secondary: MBE, R2, nRMSE and
nMAE by capacity. Report all-time and daylight-only metrics, with daylight
defined by the existing deterministic geometry mask. Add paired 7-day block
bootstrap 95% confidence intervals for B-A and C-B on the confirmatory test.

The main table contains only A, B, and C. Negative diagnostics and cache
audits go in supplementary material. Existing LCRA/FPA files remain archived
as exploratory and are not overwritten.

## Entry Point Contract

The formal runner exposes one dry-run command and one explicit full-rebuild
command:

```text
python run_experiment.py --preset fm-h16 --dry-run
python run_experiment.py --preset fm-h16 --resume --rebuild-fm
```

`--dry-run` has no model or data side effects. `--rebuild-fm` is the only
formal-runner path allowed to rebuild Chronos caches, train the locked adapter
campaign, run shuffled controls, and perform the explicit test confirmation.
