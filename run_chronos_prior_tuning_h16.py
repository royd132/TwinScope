"""Chronos-prior-only H16 tuning campaign (validation-only development).

Locked scope: PSRC, the PARA-style residual adapter, its position, the
physical tokens, the loss and the optimizer are all UNCHANGED. This
campaign only varies how the frozen Chronos-2 prior itself is generated
and read out:

  Stage A  L_C in {96,192,384,672} x readout in
           {q50, trimmed mean q20-q80, full quantile mean}  (12 configs)
           selection: PRIMARY pooled residual corr(D, R), SECONDARY
           residual-direction accuracy, TIEBREAK standalone prior RMSE;
           single-quantile q0.1..q0.9 and four-segment best-q tables are
           recorded as diagnostics only (never selection inputs).
  Stage B  raw PV input vs deterministic capacity normalization at the
           locked context/readout; lock one input contract.
  Stage C  train the unchanged PARA-style adapter (PC-FRA arm B) once per
           development station with the locked prior; promote only if
           validation RMSE improves at BOTH stations (MAE veto).

Test windows are never read, inferred, or ranked in Stages A-C. Commands::

    python run_chronos_prior_tuning_h16.py --protocol
    python run_chronos_prior_tuning_h16.py --build-caches
    python run_chronos_prior_tuning_h16.py --stage-a
    python run_chronos_prior_tuning_h16.py --stage-b
    python run_chronos_prior_tuning_h16.py --stage-c
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from formal.chronos_prior import (
    BACKEND,
    CHRONOS2_QUANTILES,
    INPUT_NORMS,
    MODEL_NAME,
    READOUT_FULL_MEAN,
    READOUT_Q50,
    READOUT_TRIMMED,
    SINGLE_Q_READOUTS,
    STAGE_A_CONTEXTS,
    STAGE_A_READOUTS,
    build_chronos2_grid_cache,
    contract_prior,
    load_grid_cache,
    per_horizon_corr,
    pool_diagnostics,
    prior_cache_path_v4,
    readout,
    residual_diagnostics,
    segment_table,
    select_stage_a,
    single_quantile_table,
    valid_starts,
)
from formal.config import JobSpec, load_best_parameters, load_experiment_config
from formal.data import load_dataset
from formal.engine import (
    _evaluate,
    _loader,
    _model_config,
    _train_model,
    build_model,
    psrc_settings,
)
from formal.fm_priors import prior_cache_path
from formal.pc_fra import (
    PcFraPack,
    build_pack,
    compute_epsilon,
    load_chronos_median,
    stratified_partners,
)

SEED = 2026
SEQ_LEN, HORIZON = 96, 16
DEV_STATIONS = ("dkasc_site31", "pvod_station02")
CONFIRM_STATIONS = ("dkasc_site9a", "pvod_station00")
HIDDEN, DROPOUT = 96, 0.1
ADAPTER_LR, ADAPTER_WD = 1e-3, 0.0
EPOCHS, PATIENCE = 50, 10
ROOT = Path("outputs/chronos_prior_tune_v1")
PCFRA_V1 = Path("outputs/pcfra_h16_v1")
SHUFFLE_SEED = 2026


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_status(root: Path, stage: str, **details) -> None:
    """Append execution state so the static protocol.json never goes stale."""
    path = root / "run_status.json"
    if path.is_file():
        status = json.loads(path.read_text(encoding="utf-8"))
    else:
        status = {"campaign": "chronos_prior_only_tune_v1",
                  "protocol_revision": 2, "history": []}
    entry = {
        "stage": stage,
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **details,
    }
    status["current_stage"] = stage
    status.setdefault("history", []).append(entry)
    _write_json(path, status)


def protocol() -> dict[str, object]:
    return {
        "campaign": "chronos_prior_only_tune_v1",
        "protocol_revision": 2,
        "status": (
            "design_specification; execution state is tracked separately in "
            "run_status.json (this file never claims designed_not_run)"
        ),
        "locked_scope": (
            "ONLY Chronos-2 prior generation/readout changes. PSRC, PARA "
            "adapter architecture/position, physical tokens, FiLM, loss "
            "and optimizer are unchanged. Chronos covariates are not used."
        ),
        "setting": "15 min; PSRC L96-H16; chronological 70/15/15; seed 2026",
        "stations": {"development_validation_only": list(DEV_STATIONS),
                      "confirmation_separate_explicit_command": list(CONFIRM_STATIONS)},
        "chronos": {
            "model": MODEL_NAME,
            "native_quantiles": list(CHRONOS2_QUANTILES),
            "psrc_context_rows": SEQ_LEN,
            "stage_a_context_rows": list(STAGE_A_CONTEXTS),
            "context_alignment": (
                "Chronos history ends at the same forecast origin as PSRC: "
                "power[s+96-L_C : s+96]"
            ),
            "readouts": {
                **{
                    name: f"native single quantile Q_{float(name[1:]):.2f}"
                    for name in SINGLE_Q_READOUTS
                },
                READOUT_Q50: "native median Q_0.5",
                READOUT_TRIMMED: "trapezoid mean of Q_tau for tau in [0.20,0.80], width-normalised",
                READOUT_FULL_MEAN: "trapezoid mean over the full grid [0.01,0.99], width-normalised",
            },
            "input_contracts": {
                "raw": "clipped nonnegative PV history, physical units",
                "capacity": (
                    "x = max(P,0)/capacity (deterministic training capacity); "
                    "forecast multiplied back by capacity exactly before clipping"
                ),
            },
            "pv_prior_contract": (
                "nonnegative; <= capacity; horizon-wise deterministic daylight "
                "mask per FUTURE row: geometric SolarElevationMask column when "
                "the dataset ships it (season/latitude aware), else fixed "
                "06:00-20:00 fallback; never a whole-H16 mask from the last "
                "historical power"
            ),
            "quantile_ordering": "21 heads sorted monotone per cell before readout",
            "recipe": "v4 caches (full 21-q grid); v3 q10/q50/q90 caches untouched",
            "windows_missing_long_context": (
                "earliest train windows with start+96-L_C<0 are excluded from "
                "that cache and from Stage C training; validation is unaffected"
            ),
        },
        "stage_a": {
            "configs": (
                "4 contexts x 12 readouts (9 single native quantiles q0.10-"
                "q0.90 + q50 + trimmed mean + full mean) = 48, no adapter "
                "training"
            ),
            "d_space": "(P_chronos - y_psrc)/target_sd",
            "r_space": "(y - y_psrc)/target_sd",
            "primary": "max pooled residual correlation corr(D, R)",
            "secondary": "max residual-direction accuracy sign(D)==sign(R)",
            "tiebreak": "min pooled standardized standalone prior RMSE",
            "diagnostics_not_selection": [
                "best q per fixed segment H1-4/H5-8/H9-12/H13-16 from "
                "{0.2,0.35,0.5,0.65,0.8} (horizon-specific table only; the "
                "formal candidate stays one global readout per config)",
                "OLS beta* and oracle projected residual RMSE",
                "per-horizon residual correlation",
            ],
        },
        "stage_b": {
            "compare": ["raw", "capacity"],
            "rule": "same pooled residual-corr criterion at the locked context/readout; tie keeps raw",
        },
        "stage_c": {
            "adapter": "unchanged PARA-style window-level H-dim adapter (PC-FRA arm B)",
            "hyperparameters": {
                "hidden": HIDDEN, "dropout": DROPOUT, "lr": ADAPTER_LR,
                "weight_decay": ADAPTER_WD, "epochs": EPOCHS, "patience": PATIENCE,
            },
            "epsilon": "training p90 |y-y_psrc| capped at 0.2*capacity/sd (unchanged rule)",
            "promotion": (
                "B improves validation RMSE at BOTH stations; rejected if MAE "
                "worsens at BOTH; on failure the line stops with no retuning"
            ),
        },
        "test_policy": "test is never read, generated, dumped or ranked in stages A-C",
        "shuffle_diagnostics": "not rerun (unchanged from pcfra_h16_v1 conclusion)",
    }


def _checkpoint(station: str) -> Path:
    return Path("outputs/final_h16") / (
        f"{station}_L{SEQ_LEN}_H{HORIZON}_psrc_seed{SEED}"
    ) / "checkpoint.pt"


def _load_bundle(station, config):
    return load_dataset(
        config.dataset(station).path,
        SEQ_LEN, HORIZON, config.train_fraction, config.validation_end_fraction,
    )


def _build_frozen(station, config, bundle, params, state, device):
    job = JobSpec(station, "psrc", SEQ_LEN, HORIZON, SEED)
    model = build_model(
        job.model,
        _model_config(job, bundle, params, bool(config.psrc["gate"])),
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, job


def _infer(model, bundle, starts, batch, device):
    loader = _loader(bundle, starts, batch, False, SEED)
    return _evaluate(model, loader, bundle, device, collect=True)


# --- frozen PSRC predictions (validation + training) ------------------------

def psrc_arrays(station, config, records, bundle, starts):
    """Physical/standardized frozen PSRC outputs and targets for windows."""
    params = dict(records[station]["params"])
    state = torch.load(
        _checkpoint(station), map_location="cpu", weights_only=True
    )["state_dict"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frozen, _ = _build_frozen(station, config, bundle, params, state, device)
    _, arrays = _infer(frozen, bundle, starts, int(params.get("batch", 128)), device)
    sd, mu = float(bundle.feature_sd[-1]), float(bundle.feature_mu[-1])
    return {
        "pred_std": arrays["pred_std"],
        "target_std": arrays["target_std"],
        "pred_phys": arrays["pred_std"] * sd + mu,
        "target_phys": arrays["target_std"] * sd + mu,
        "target_sd": sd,
    }


# --- Stage A/B prior analysis -----------------------------------------------

def _config_metrics(station, bundle, q_grid, taus, readout_name, psrc):
    prior_raw = readout(q_grid, taus, readout_name)
    prior = contract_prior(
        prior_raw, bundle, bundle.validation.starts, SEQ_LEN, HORIZON
    )
    diag = residual_diagnostics(
        prior, psrc["pred_phys"], psrc["target_phys"], psrc["target_sd"]
    )
    diag["per_horizon_residual_corr"] = per_horizon_corr(
        prior, psrc["pred_phys"], psrc["target_phys"], psrc["target_sd"]
    )
    d = ((prior - psrc["pred_phys"]) / psrc["target_sd"]).reshape(-1)
    r = ((psrc["target_phys"] - psrc["pred_phys"]) / psrc["target_sd"]).reshape(-1)
    e = ((prior - psrc["target_phys"]) / psrc["target_sd"]).reshape(-1)
    usable = np.abs(d) > 1e-6
    arrays = {
        "d": d, "r": r, "err_std": e,
        "dir_hits": int(np.sum(np.sign(d[usable]) == np.sign(r[usable]))),
        "dir_n": int(np.sum(usable)),
    }
    return diag, arrays, prior_raw, prior


def _evaluate_norm(station, config, records, bundle, psrc, context_rows, input_norm):
    """All readouts for one (station, context, norm), split validation only."""
    path = prior_cache_path_v4(bundle.path, HORIZON, context_rows, input_norm)
    starts, q_grid, meta = load_grid_cache(path)
    if not np.array_equal(starts, np.sort(starts)):
        raise ValueError("cache starts unsorted")
    taus = np.asarray(CHRONOS2_QUANTILES)
    # Validation windows must all be covered (they are far from row 0).
    pos = np.searchsorted(starts, bundle.validation.starts)
    if not np.all(starts[pos] == bundle.validation.starts):
        raise ValueError(f"{path} does not cover every validation window")
    val_q = q_grid[pos]
    out = {"_meta": meta}
    pooled_arrays = {}
    for name in STAGE_A_READOUTS:
        diag, arrays, prior_raw, prior = _config_metrics(
            station, bundle, val_q, taus, name, psrc
        )
        out[name] = diag
        pooled_arrays[name] = arrays
    # Diagnostics-only tables (cheap; never enter selection).
    def contract_fn(raw):
        return contract_prior(
            raw, bundle, bundle.validation.starts, SEQ_LEN, HORIZON
        )
    out["single_quantile_diagnostics"] = single_quantile_table(
        val_q, taus, contract_fn, psrc["pred_phys"], psrc["target_phys"],
        psrc["target_sd"],
    )
    out["segment_diagnostics"] = segment_table(
        val_q, taus, contract_fn, psrc["pred_phys"], psrc["target_phys"],
        psrc["target_sd"],
    )
    return out, pooled_arrays, val_q


def stage_a(config, records, root: Path) -> dict[str, object]:
    """Evaluate the 48 registered configs on validation windows only."""
    report_path = root / "stage_a_report.json"
    decision_path = root / "stage_a_decision.json"
    per_station: dict[str, dict] = {}
    pooled_inputs: dict[tuple[int, str], dict[str, dict]] = {}
    for station in DEV_STATIONS:
        bundle = _load_bundle(station, config)
        psrc = psrc_arrays(station, config, records, bundle, bundle.validation.starts)
        station_block: dict[str, object] = {}
        for ctx in STAGE_A_CONTEXTS:
            block, arrays, val_q = _evaluate_norm(
                station, config, records, bundle, psrc, ctx, "raw"
            )
            station_block[f"ctx{ctx}"] = block
            for name in STAGE_A_READOUTS:
                pooled_inputs[(ctx, name)] = pooled_inputs.get((ctx, name), {})
                pooled_inputs[(ctx, name)][station] = {"_arrays": arrays[name]}
        # Audit: v4 ctx96/q50 must reproduce the frozen v3 q50 cache.
        v3_path = prior_cache_path(bundle.path, BACKEND, HORIZON, recipe="v3")
        v3_starts, v3_q50, _ = load_chronos_median(v3_path, bundle.path)
        pos3 = np.searchsorted(v3_starts, bundle.validation.starts)
        v4_q50 = readout(val_q, np.asarray(CHRONOS2_QUANTILES), READOUT_Q50)
        delta = np.abs(v4_q50 - v3_q50[pos3].astype(np.float32))
        station_block["v3_vs_v4_ctx96_q50_audit"] = {
            "max_abs_diff_physical": float(delta.max()),
            "mean_abs_diff_physical": float(delta.mean()),
            "note": "differences can only come from 21-level vs 3-level monotone sort",
        }
        per_station[station] = station_block

    rows = []
    for ctx in STAGE_A_CONTEXTS:
        for name in STAGE_A_READOUTS:
            pooled = pool_diagnostics(pooled_inputs[(ctx, name)], DEV_STATIONS)
            row = {"context_rows": int(ctx), "readout": name, **pooled}
            rows.append(row)
    decision = select_stage_a(rows)
    report = {
        "per_station": per_station,
        "pooled_rows": [
            {k: v for k, v in row.items()} for row in rows
        ],
    }
    _write_json(decision_path, decision)
    _write_json(report_path, report)
    return {"decision": decision, "report": report}


def stage_b(config, records, root: Path) -> dict[str, object]:
    """Compare raw vs capacity normalization at the locked context."""
    decision_a = json.loads(
        (root / "stage_a_decision.json").read_text(encoding="utf-8")
    )
    ctx = int(decision_a["selected_context_rows"])
    readout_name = str(decision_a["selected_readout"])
    pooled_inputs: dict[str, dict[str, dict]] = {}
    per_station: dict[str, dict] = {}
    for station in DEV_STATIONS:
        bundle = _load_bundle(station, config)
        psrc = psrc_arrays(station, config, records, bundle, bundle.validation.starts)
        per_station[station] = {}
        for norm in INPUT_NORMS:
            path = prior_cache_path_v4(bundle.path, HORIZON, ctx, norm)
            if not path.is_file():
                # The capacity cache is first needed here; it is generated
                # only for the locked context, train+validation windows.
                starts = np.sort(np.concatenate(
                    [bundle.train.starts, bundle.validation.starts]
                ))
                build_chronos2_grid_cache(
                    bundle, starts, HORIZON, ctx, norm,
                    seq_len=SEQ_LEN, batch_size=128,
                )
            starts, q_grid, _ = load_grid_cache(path)
            pos = np.searchsorted(starts, bundle.validation.starts)
            if not np.all(starts[pos] == bundle.validation.starts):
                raise ValueError(f"{path} does not cover every validation window")
            prior_raw = readout(q_grid[pos], np.asarray(CHRONOS2_QUANTILES), readout_name)
            prior = contract_prior(
                prior_raw, bundle, bundle.validation.starts, SEQ_LEN, HORIZON
            )
            diag = residual_diagnostics(
                prior, psrc["pred_phys"], psrc["target_phys"], psrc["target_sd"]
            )
            d = ((prior - psrc["pred_phys"]) / psrc["target_sd"]).reshape(-1)
            r = ((psrc["target_phys"] - psrc["pred_phys"]) / psrc["target_sd"]).reshape(-1)
            e = ((prior - psrc["target_phys"]) / psrc["target_sd"]).reshape(-1)
            usable = np.abs(d) > 1e-6
            per_station[station][norm] = diag
            pooled_inputs[norm] = pooled_inputs.get(norm, {})
            pooled_inputs[norm][station] = {
                "_arrays": {
                    "d": d, "r": r, "err_std": e,
                    "dir_hits": int(np.sum(np.sign(d[usable]) == np.sign(r[usable]))),
                    "dir_n": int(np.sum(usable)),
                }
            }
    pooled = {
        norm: pool_diagnostics(pooled_inputs[norm], DEV_STATIONS)
        for norm in INPUT_NORMS
    }
    raw_corr = pooled["raw"]["pooled_residual_corr"]
    cap_corr = pooled["capacity"]["pooled_residual_corr"]
    # Cached priors are fp16; only a corr margin far above storage noise is
    # treated as a real improvement (Stage A meaningful margins are >= 1e-2).
    chosen = "capacity" if cap_corr > raw_corr + 1e-6 else "raw"
    lock = {
        "locked_context_rows": ctx,
        "locked_readout": readout_name,
        "locked_input_norm": chosen,
        "selection_rule": (
            "max pooled residual corr among {raw, capacity}; ties keep raw"
        ),
        "pooled": pooled,
        "per_station": per_station,
    }
    _write_json(root / "stage_b_lock.json", lock)
    return lock


# --- Stage C: one unchanged PARA adapter per station ------------------------

def _pc_cfg(epsilon: float) -> dict[str, object]:
    return {
        "horizon": HORIZON, "variant": "para", "hidden": HIDDEN,
        "dropout": DROPOUT, "film_bound": 0.1, "epsilon": float(epsilon),
        "level_mu": 0.0, "level_sd": 1.0, "sigma_mu": 0.0, "sigma_sd": 1.0,
    }


def _merge_packs(train_pack, val_pack):
    starts = np.concatenate([train_pack.starts, val_pack.starts])
    vectors = np.concatenate([train_pack.embeddings, val_pack.embeddings])
    partners = np.concatenate([train_pack.partner_starts, val_pack.partner_starts])
    return PcFraPack(starts, vectors, partners, HORIZON)


def stage_c(config, records, root: Path) -> dict[str, object]:
    lock_b = json.loads((root / "stage_b_lock.json").read_text(encoding="utf-8"))
    ctx = int(lock_b["locked_context_rows"])
    norm = str(lock_b["locked_input_norm"])
    readout_name = str(lock_b["locked_readout"])
    taus = np.asarray(CHRONOS2_QUANTILES)
    table = {}
    for station in DEV_STATIONS:
        station_dir = root / "stage_c" / station
        val_path_done = station_dir / "validation.json"
        if val_path_done.is_file():
            table[station] = json.loads(val_path_done.read_text(encoding="utf-8"))
            continue
        bundle = _load_bundle(station, config)
        params = dict(records[station]["params"])
        state = torch.load(
            _checkpoint(station), map_location="cpu", weights_only=True
        )["state_dict"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        frozen, job = _build_frozen(station, config, bundle, params, state, device)
        batch = int(params.get("batch", 128))
        cache_path = prior_cache_path_v4(bundle.path, HORIZON, ctx, norm)
        cache_starts, q_grid, cache_meta = load_grid_cache(cache_path)
        grid_lookup = {int(s): i for i, s in enumerate(cache_starts)}

        # Only windows covered by the locked cache (long-context early
        # training windows are excluded consistently).
        train_starts = valid_starts(bundle.train.starts, SEQ_LEN, ctx)
        val_starts = bundle.validation.starts
        train_a = _infer(frozen, bundle, train_starts, batch, device)[1]
        val_a = _infer(frozen, bundle, val_starts, batch, device)[1]
        sd, mu = float(bundle.feature_sd[-1]), float(bundle.feature_mu[-1])

        def prior_lookup(starts_, arrays):
            return {
                int(s): readout(
                    q_grid[grid_lookup[int(s)]][None], taus, readout_name
                )[0]
                for s in starts_
            }

        median_lookup = prior_lookup(train_starts, train_a)
        median_lookup.update(prior_lookup(val_starts, val_a))
        psrc_lookup = {
            int(s): train_a["pred_std"][i] for i, s in enumerate(train_starts)
        }
        psrc_lookup.update(
            {int(s): val_a["pred_std"][i] for i, s in enumerate(val_starts)}
        )
        # Training-only stats/epsilon on the actually-used windows.
        level = []
        sigma = []
        from formal.pc_fra import window_scalar_features
        for s in train_starts:
            lr, sg, _, _, _, _ = window_scalar_features(
                bundle, int(s), SEQ_LEN, HORIZON
            )
            level.append(lr)
            sigma.append(sg)
        stats = {
            "level_mu": float(np.mean(level)),
            "level_sd": float(np.std(level) + 1e-8),
            "sigma_mu": float(np.mean(sigma)),
            "sigma_sd": float(np.std(sigma) + 1e-8),
        }
        epsilon_info = compute_epsilon(
            train_a["pred_std"] - train_a["target_std"], sd, bundle.capacity
        )
        cfg = _pc_cfg(epsilon_info["epsilon"])
        cfg.update({k: float(v) for k, v in stats.items() if k in cfg})

        train_partners, _ = stratified_partners(
            train_starts, bundle, SEQ_LEN, HORIZON, SHUFFLE_SEED
        )
        val_partners, _ = stratified_partners(
            val_starts, bundle, SEQ_LEN, HORIZON, SHUFFLE_SEED + 1
        )
        train_pack = build_pack(
            bundle, train_starts, SEQ_LEN, HORIZON,
            median_lookup, psrc_lookup, train_partners,
        )
        val_pack = build_pack(
            bundle, val_starts, SEQ_LEN, HORIZON,
            median_lookup, psrc_lookup, val_partners,
        )
        run_params = copy.deepcopy(params)
        run_params.update({"lr": ADAPTER_LR, "weight_decay": ADAPTER_WD,
                            "scheduler": "none"})
        # Engine enumerates training windows from bundle.train.starts; pass a
        # shallow in-memory copy restricted to long-context-covered windows so
        # every requested start has a pack row (engine stays frozen).
        import dataclasses
        n_total_train = int(len(bundle.train.starts))
        train_bundle = dataclasses.replace(
            bundle,
            train=dataclasses.replace(bundle.train, starts=train_starts),
        )
        started = time.perf_counter()
        model, history, stopper, _ = _train_model(
            job, config, train_bundle, run_params, psrc_settings(params, config.psrc),
            epochs=EPOCHS, patience=PATIENCE, fm_embeddings=_merge_packs(train_pack, val_pack),
            initial_state_dict=state, train_pc_only=True, pc_fra_config=cfg,
        )
        metrics_b, arrays_b = _evaluate(
            model, _loader(bundle, val_starts, batch, False, SEED, fm_embeddings=val_pack),
            bundle, device, collect=True,
        )
        station_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": model.pc_fra_adapter.state_dict()},
            station_dir / "adapter_checkpoint.pt",
        )
        pred_b_phys = np.maximum(0.0, arrays_b["pred_std"] * sd + mu)
        target_phys = arrays_b["target_std"] * sd + mu
        pred_a_phys = np.maximum(0.0, val_a["pred_std"] * sd + mu)
        from formal.metrics import regression_metrics

        ma = regression_metrics(pred_a_phys, target_phys, bundle.capacity)
        mb = regression_metrics(pred_b_phys, target_phys, bundle.capacity)
        payload = {
            "station": station,
            "locked": {"context_rows": ctx, "input_norm": norm,
                        "readout": readout_name},
            "n_train_windows_used": int(len(train_starts)),
            "n_train_windows_dropped_long_context": int(
                n_total_train - len(train_starts)
            ),
            "epsilon": epsilon_info,
            "training_statistics": stats,
            "best_epoch": stopper.best_epoch,
            "epochs_ran": len(history),
            "seconds": time.perf_counter() - started,
            "A_frozen_psrc": {
                "rmse_physical": float(ma["rmse_physical"]),
                "mae_physical": float(ma["mae_physical"]),
                "mbe_physical": float(ma["mbe_physical"]),
                "r2": float(ma["r2"]),
            },
            "B_para_locked_prior": {
                "rmse_physical": float(mb["rmse_physical"]),
                "mae_physical": float(mb["mae_physical"]),
                "mbe_physical": float(mb["mbe_physical"]),
                "r2": float(mb["r2"]),
            },
        }
        _write_json(station_dir / "validation.json", payload)
        table[station] = payload

    b_rmse_gain = {
        s: table[s]["A_frozen_psrc"]["rmse_physical"]
        - table[s]["B_para_locked_prior"]["rmse_physical"]
        for s in DEV_STATIONS
    }
    b_mae_gain = {
        s: table[s]["A_frozen_psrc"]["mae_physical"]
        - table[s]["B_para_locked_prior"]["mae_physical"]
        for s in DEV_STATIONS
    }
    b_promotes = all(b_rmse_gain[s] > 0 for s in DEV_STATIONS) and not all(
        b_mae_gain[s] < 0 for s in DEV_STATIONS
    )
    decision = {
        "locked_prior": {
            "context_rows": ctx, "input_norm": norm, "readout": readout_name,
        },
        "per_station": {
            s: {
                "A_rmse": table[s]["A_frozen_psrc"]["rmse_physical"],
                "B_rmse": table[s]["B_para_locked_prior"]["rmse_physical"],
                "A_mae": table[s]["A_frozen_psrc"]["mae_physical"],
                "B_mae": table[s]["B_para_locked_prior"]["mae_physical"],
                "best_epoch": table[s]["best_epoch"],
            }
            for s in DEV_STATIONS
        },
        "per_station_rmse_gain_B_minus_A": b_rmse_gain,
        "per_station_mae_gain_B_minus_A": b_mae_gain,
        "B_promotes": bool(b_promotes),
        "test_evaluated": False,
        "next_step_if_promoted": (
            "explicit confirmation command on dkasc_site9a/pvod_station00 "
            "with frozen config, evaluated once"
        ),
    }
    _write_json(root / "stage_c" / "stage_c_decision.json", decision)
    return decision


def build_caches(config, records, root: Path, contexts, norms) -> None:
    manifest = []
    for station in DEV_STATIONS:
        bundle = _load_bundle(station, config)
        starts = np.sort(
            np.concatenate([bundle.train.starts, bundle.validation.starts])
        )
        for ctx in contexts:
            for norm in norms:
                path = build_chronos2_grid_cache(
                    bundle, starts, HORIZON, int(ctx), norm,
                    seq_len=SEQ_LEN, batch_size=128,
                )
                kept = len(valid_starts(starts, SEQ_LEN, int(ctx)))
                manifest.append({
                    "station": station, "context_rows": int(ctx),
                    "input_norm": norm, "path": str(path),
                    "n_windows": kept,
                    "n_dropped_short_context": int(len(starts) - kept),
                })
                print(
                    f"[cache] {station} ctx={ctx} norm={norm}: "
                    f"{kept} windows ({len(starts) - kept} dropped)",
                    flush=True,
                )
    _write_json(root / "cache_manifest.json", {"caches": manifest})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/formal_experiment.json"))
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--protocol", action="store_true")
    parser.add_argument("--build-caches", action="store_true")
    parser.add_argument("--stage-a", action="store_true")
    parser.add_argument("--stage-b", action="store_true")
    parser.add_argument("--stage-c", action="store_true")
    args = parser.parse_args(argv)

    spec = protocol()
    if args.protocol or not any(
        (args.build_caches, args.stage_a, args.stage_b, args.stage_c)
    ):
        print(json.dumps(spec, ensure_ascii=False, indent=2))
        if not any((args.build_caches, args.stage_a, args.stage_b, args.stage_c)):
            print("[chronos-prior] protocol only; no data loaded, no files written.")
        return 0

    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "protocol.json", spec)
    config = load_experiment_config(args.config)
    records = load_best_parameters(config.best_parameters)

    if args.build_caches:
        build_caches(config, records, root, STAGE_A_CONTEXTS, ("raw",))
        _write_status(root, "caches_built",
                      contexts=list(STAGE_A_CONTEXTS), norms=["raw"])
    if args.stage_a:
        result = stage_a(config, records, root)
        d = result["decision"]
        print(
            f"[stage-a] locked L_C={d['selected_context_rows']} "
            f"readout={d['selected_readout']} "
            f"corr={d['selected_metrics']['pooled_residual_corr']:.4f} "
            f"dir={d['selected_metrics']['pooled_direction_accuracy']:.4f}",
            flush=True,
        )
        _write_status(
            root, "stage_a_completed",
            context_rows=d["selected_context_rows"],
            readout=d["selected_readout"],
            pooled_residual_corr=d["selected_metrics"]["pooled_residual_corr"],
            pooled_direction_accuracy=d["selected_metrics"]["pooled_direction_accuracy"],
            n_configs=4 * len(STAGE_A_READOUTS),
        )
    if args.stage_b:
        lock = stage_b(config, records, root)
        print(
            f"[stage-b] locked input_norm={lock['locked_input_norm']} "
            f"(raw corr={lock['pooled']['raw']['pooled_residual_corr']:.4f}, "
            f"capacity corr={lock['pooled']['capacity']['pooled_residual_corr']:.4f})",
            flush=True,
        )
        _write_status(
            root, "stage_b_completed", input_norm=lock["locked_input_norm"],
            context_rows=lock["locked_context_rows"],
            readout=lock["locked_readout"],
        )
    if args.stage_c:
        decision = stage_c(config, records, root)
        print(
            "[stage-c] B promotes:", decision["B_promotes"],
            "| rmse gains:", decision["per_station_rmse_gain_B_minus_A"],
            flush=True,
        )
        _write_status(
            root, "stage_c_completed",
            b_promotes=bool(decision["B_promotes"]),
            test_evaluated=False,
            per_station=decision.get("per_station", {}),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
