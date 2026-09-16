"""Strict L96-H16 Chronos prior calibration campaign (validation only).

Context is LOCKED to 96 for BOTH PSRC and Chronos - identical information
set X_{t-95:t}; no 192/384/672 windows, no covariates, no extra data. Only
how the frozen Chronos-2 distributional output is read and how its
disagreement with the frozen PSRC is scaled may change. The late PARA
adapter, its position, physical tokens, FiLM, loss and optimizer never
change.

Phase A (no training): 6 fixed readouts (q35/q40/q45/q50/trimmed/full
mean) + 5 skew-shape trajectories P*=q50+beta*(q90+q10-2*q50), all four
stations' VALIDATION windows. Selection is worst-site (max-min station
residual correlation), never pooled; pick the 2 strongest candidates.

Phase B (no training): for those 2 candidates, disagreement-scale rules
P'=y_psrc+gamma*(P*-y_psrc): gamma in {.1,.25,.5,.75,1} plus two fixed
monotonic 4-segment schedules. Scalar scaling cannot change rho or
direction (asserted); it only moves kappa=sigma_D/sigma_R toward 1.
Lock one scale rule per candidate.

Phase C: train the unchanged PARA late adapter once per locked prior on
the two development stations, evaluate validation RMSE/MAE against the
frozen PSRC arm. Test is never read for any station.

Stages: --protocol / --build-caches / --phase-a / --phase-b / --phase-c.
"""

from __future__ import annotations

import copy
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from formal.chronos_prior import (
    CHRONOS2_QUANTILES,
    HORIZON_SEGMENTS,
    L96_PHASE_A_READOUTS,
    SCALE_GAMMAS,
    SEGMENT_GAMMA_SCHEDULES,
    SKEW_BETAS,
    build_chronos2_grid_cache,
    contract_prior,
    load_grid_cache,
    per_horizon_corr,
    prior_cache_path_v4,
    readout,
    residual_diagnostics,
    scale_disagreement,
    scale_disagreement_segments,
    select_worst_site,
    skew_readout,
    skew_readout_name,
)
from formal.config import load_best_parameters, load_experiment_config
from formal.pc_fra import PcFraPack, build_pack, stratified_partners, window_scalar_features
from formal.pc_fra import compute_epsilon

import run_chronos_prior_tuning_h16 as prev
from formal.engine import _evaluate, _loader, _train_model, psrc_settings

# Reuse every frozen constant / helper from the registered campaign.
SEQ_LEN = prev.SEQ_LEN
HORIZON = prev.HORIZON
SEED = prev.SEED
SHUFFLE_SEED = prev.SHUFFLE_SEED
HIDDEN = prev.HIDDEN
DROPOUT = prev.DROPOUT
EPOCHS = prev.EPOCHS
PATIENCE = prev.PATIENCE
ADAPTER_LR = prev.ADAPTER_LR
ADAPTER_WD = prev.ADAPTER_WD

ROOT = Path("outputs/chronos_l96_calib_v1")
LOCKED_CONTEXT = 96
LOCKED_NORM = "raw"
# Prior calibration uses all four stations of the formal comparison subset
# (outputs/final_h16/formal_comparison_subset.xlsx): DKASC 31/1A, PVOD 02,
# HKUST. Adapter training (Phase C) stays on the two development stations.
ALL_STATIONS: tuple[str, ...] = (
    "dkasc_site31", "pvod_station02", "dkasc_site1a", "hkust",
)
DEV_STATIONS: tuple[str, ...] = ("dkasc_site31", "pvod_station02")


# --- utilities --------------------------------------------------------------

def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_status(root: Path, stage: str, **details) -> None:
    path = root / "run_status.json"
    if path.is_file():
        status = json.loads(path.read_text(encoding="utf-8"))
    else:
        status = {"campaign": "chronos_l96_calib_v1", "history": []}
    status["current_stage"] = stage
    status.setdefault("history", []).append(
        {"stage": stage,
         "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         **details}
    )
    _write_json(path, status)


def protocol() -> dict[str, object]:
    return {
        "campaign": "chronos_l96_calib_v1",
        "status": (
            "design_specification; execution state lives in run_status.json"
        ),
        "hard_lock": {
            "context_rows_PSRC": SEQ_LEN,
            "context_rows_chronos": LOCKED_CONTEXT,
            "horizon": HORIZON,
            "input_contract": LOCKED_NORM,
            "information_set": "X_C = X_P = X_{t-95:t} (identical windows)",
            "forbidden": [
                "L192/L384/L672 and any earlier history",
                "future NWP / covariates / extra data",
                "adapter architecture, position, tokens, FiLM, loss, "
                "optimizer changes",
            ],
        },
        "stations": {
            "all_used_in_prior_selection_validation_only": list(ALL_STATIONS),
            "phase_c_adapter_training": list(DEV_STATIONS),
            "note": (
                "dkasc_site1a/hkust validation participates in worst-site "
                "prior selection (they are the formal comparison subset, not "
                "independent confirmation stations); all TEST windows remain "
                "untouched"
            ),
        },
        "phase_a": {
            "readouts": list(L96_PHASE_A_READOUTS),
            "skew_trajectories": {
                skew_readout_name(b): (
                    f"P*=q50+({b:+.2f})*(q90+q10-2*q50)"
                )
                for b in SKEW_BETAS
            },
            "n_candidates": len(L96_PHASE_A_READOUTS) + len(SKEW_BETAS),
            "metrics_per_station": [
                "rho=corr(D,R)", "direction sign(D)==sign(R)",
                "kappa=sigma_D/sigma_R", "MBE", "per-horizon rho",
            ],
            "selection": (
                "max_theta min_s rho_s; secondary max min_s direction; "
                "tiebreak min max_s |kappa_s-1|; keep top 2"
            ),
        },
        "phase_b": {
            "scalar_gammas": list(SCALE_GAMMAS),
            "segment_schedules": SEGMENT_GAMMA_SCHEDULES,
            "formula": "P'=y_psrc+gamma*(P*-y_psrc)",
            "scale_invariance": (
                "scalar gamma>0 leaves rho and direction unchanged "
                "(asserted); only kappa and MBE move"
            ),
            "selection": (
                "per candidate: all stations rho>0, then minimize max_s "
                "|kappa_s-1|; segment schedules evaluated directly"
            ),
        },
        "phase_c": {
            "adapter": "unchanged PARA late residual adapter (one train per prior/station)",
            "hyperparameters": {
                "hidden": HIDDEN, "dropout": DROPOUT, "lr": ADAPTER_LR,
                "weight_decay": ADAPTER_WD, "epochs": EPOCHS,
                "patience": PATIENCE, "seed": SEED,
            },
            "promotion": (
                "per locked prior: validation RMSE improves at BOTH "
                "development stations; MAE must not worsen at both"
            ),
        },
        "test_policy": "test is never read, generated, dumped or ranked",
    }


# --- station data loading ---------------------------------------------------

def _val_data(station, config, records):
    """Bundle, frozen PSRC validation arrays, ctx96 validation q-grid."""
    bundle = prev._load_bundle(station, config)
    psrc = prev.psrc_arrays(
        station, config, records, bundle, bundle.validation.starts
    )
    path = prior_cache_path_v4(bundle.path, HORIZON, LOCKED_CONTEXT, LOCKED_NORM)
    starts, q_grid, meta = load_grid_cache(path)
    pos = np.searchsorted(starts, bundle.validation.starts)
    if not np.all(starts[pos] == bundle.validation.starts):
        raise ValueError(f"{path.name} does not cover {station} validation")
    return bundle, psrc, q_grid[pos], meta


def _candidate_prior(q_grid, taus, bundle, name):
    """Contracted physical prior trajectory for a Phase A candidate name."""
    if name.startswith("skew_b"):
        beta = float(name[len("skew_b"):])
        raw = skew_readout(q_grid, taus, beta)
    else:
        raw = readout(q_grid, taus, name)
    return contract_prior(raw, bundle, bundle.validation.starts, SEQ_LEN, HORIZON)


def _diag(bundle, prior, psrc):
    diag = residual_diagnostics(
        prior, psrc["pred_phys"], psrc["target_phys"], psrc["target_sd"]
    )
    diag["per_horizon_residual_corr"] = per_horizon_corr(
        prior, psrc["pred_phys"], psrc["target_phys"], psrc["target_sd"]
    )
    return diag


# --- caches -----------------------------------------------------------------

def build_caches(config, records, root: Path) -> None:
    manifest = []
    for station in ALL_STATIONS:
        bundle = prev._load_bundle(station, config)
        starts = np.sort(
            np.concatenate([bundle.train.starts, bundle.validation.starts])
        )
        path = build_chronos2_grid_cache(
            bundle, starts, HORIZON, LOCKED_CONTEXT, LOCKED_NORM,
            seq_len=SEQ_LEN, batch_size=128,
        )
        kept = len(starts)  # L96 == seq_len: no window is dropped
        manifest.append({
            "station": station, "context_rows": LOCKED_CONTEXT,
            "input_norm": LOCKED_NORM, "path": str(path),
            "n_windows": int(kept), "n_dropped_short_context": 0,
        })
        print(f"[cache] {station} ctx=96 norm=raw: {kept} windows", flush=True)
    _write_json(root / "cache_manifest.json", {"caches": manifest})


# --- Phase A ----------------------------------------------------------------

PHASE_A_CANDIDATES = tuple(L96_PHASE_A_READOUTS) + tuple(
    skew_readout_name(b) for b in SKEW_BETAS
)


def phase_a(config, records, root: Path) -> dict[str, object]:
    taus = np.asarray(CHRONOS2_QUANTILES)
    rows, report = [], {}
    for station in ALL_STATIONS:
        bundle, psrc, val_q, _ = _val_data(station, config, records)
        report[station] = {}
        for name in PHASE_A_CANDIDATES:
            prior = _candidate_prior(val_q, taus, bundle, name)
            report[station][name] = _diag(bundle, prior, psrc)
    for name in PHASE_A_CANDIDATES:
        rows.append({
            "candidate": name,
            "per_station": {s: report[s][name] for s in ALL_STATIONS},
        })
    decision = select_worst_site(rows, ALL_STATIONS, top_k=2)
    decision["context_rows"] = LOCKED_CONTEXT
    decision["input_norm"] = LOCKED_NORM
    _write_json(root / "phase_a_report.json", {"per_station": report})
    _write_json(root / "phase_a_decision.json", decision)
    return decision


# --- Phase B ----------------------------------------------------------------

def _scale_rules():
    rules = [("scalar", f"gamma{g:.2f}", float(g)) for g in SCALE_GAMMAS]
    for nm, groups in SEGMENT_GAMMA_SCHEDULES.items():
        rules.append(("segments", nm, tuple(float(g) for g in groups)))
    return rules


def _apply_rule(prior, psrc_phys, rule):
    kind, _, value = rule
    if kind == "scalar":
        return scale_disagreement(prior, psrc_phys, value)
    return scale_disagreement_segments(prior, psrc_phys, value)


def phase_b(config, records, root: Path) -> dict[str, object]:
    decision_a = json.loads(
        (root / "phase_a_decision.json").read_text(encoding="utf-8")
    )
    chosen = [item["candidate"] for item in decision_a["selected"]]
    taus = np.asarray(CHRONOS2_QUANTILES)
    rules = _scale_rules()

    # Recompute the contracted Phase A prior per station once per candidate.
    base = {}
    for station in ALL_STATIONS:
        bundle, psrc, val_q, _ = _val_data(station, config, records)
        base[station] = {"bundle": bundle, "psrc": psrc, "priors": {}}
        for name in chosen:
            base[station]["priors"][name] = _candidate_prior(
                val_q, taus, bundle, name
            )

    report, locks = {}, []
    for name in chosen:
        rule_rows, per_station_report = [], {}
        for rule in rules:
            per_station = {}
            max_corr_shift = 0.0
            max_dir_shift = 0.0
            for station in ALL_STATIONS:
                ctx = base[station]
                p0 = ctx["priors"][name]
                p1 = _apply_rule(p0, ctx["psrc"]["pred_phys"], rule)
                d0 = _diag(ctx["bundle"], p0, ctx["psrc"])
                d1 = _diag(ctx["bundle"], p1, ctx["psrc"])
                per_station[station] = d1
                if rule[0] == "scalar":
                    max_corr_shift = max(
                        max_corr_shift,
                        abs(d1["residual_corr"] - d0["residual_corr"]),
                    )
                    max_dir_shift = max(
                        max_dir_shift,
                        abs(d1["direction_accuracy"] - d0["direction_accuracy"]),
                    )
                    # Scalar rescaling by gamma>0 is a positive affine
                    # transform of D: corr is an exact invariant. Direction is
                    # invariant too apart from rows re-crossing the |D|>eps
                    # usability threshold, which is a metric-set edge effect.
                    assert abs(
                        d1["residual_corr"] - d0["residual_corr"]
                    ) < 1e-5, (name, rule, station)
                    assert abs(
                        d1["direction_accuracy"] - d0["direction_accuracy"]
                    ) < 5e-3, (name, rule, station)
            all_positive = all(
                per_station[s]["residual_corr"] > 0.0 for s in ALL_STATIONS
            )
            worst_gap = max(
                abs(per_station[s]["scale_ratio_kappa"] - 1.0)
                for s in ALL_STATIONS
            )
            rule_rows.append({
                "rule_kind": rule[0], "rule_name": rule[1], "rule_value": rule[2],
                "all_stations_rho_positive": all_positive,
                "worst_station_kappa_gap": float(worst_gap),
                "max_scalar_invariance_corr_shift": float(max_corr_shift),
                "max_scalar_invariance_dir_shift": float(max_dir_shift),
                "per_station": per_station,
            })
            per_station_report[rule[1]] = per_station

        feasible = [r for r in rule_rows if r["all_stations_rho_positive"]]
        pool = feasible or rule_rows
        best = min(pool, key=lambda r: r["worst_station_kappa_gap"])
        locks.append({
            "candidate": name,
            "scale_rule_kind": best["rule_kind"],
            "scale_rule_name": best["rule_name"],
            "scale_rule_value": list(best["rule_value"])
            if isinstance(best["rule_value"], tuple) else best["rule_value"],
            "all_stations_rho_positive": best["all_stations_rho_positive"],
            "worst_station_kappa_gap": best["worst_station_kappa_gap"],
            "per_station": {
                s: {
                    "residual_corr": best["per_station"][s]["residual_corr"],
                    "direction_accuracy":
                        best["per_station"][s]["direction_accuracy"],
                    "scale_ratio_kappa":
                        best["per_station"][s]["scale_ratio_kappa"],
                    "prior_mbe_physical":
                        best["per_station"][s]["prior_mbe_physical"],
                    "prior_rmse_physical":
                        best["per_station"][s]["prior_rmse_physical"],
                }
                for s in ALL_STATIONS
            },
        })
        report[name] = {
            "rules": [
                {
                    "rule_name": r["rule_name"],
                    "all_stations_rho_positive": r["all_stations_rho_positive"],
                    "worst_station_kappa_gap": r["worst_station_kappa_gap"],
                    "per_station_kappa": {
                        s: r["per_station"][s]["scale_ratio_kappa"]
                        for s in ALL_STATIONS
                    },
                    "per_station_corr": {
                        s: r["per_station"][s]["residual_corr"]
                        for s in ALL_STATIONS
                    },
                }
                for r in rule_rows
            ]
        }

    out = {
        "context_rows": LOCKED_CONTEXT, "input_norm": LOCKED_NORM,
        "locked_priors": locks,
        "selection_rule": (
            "all four stations rho>0 required; among feasible rules pick "
            "min max_s |kappa_s-1|"
        ),
    }
    _write_json(root / "phase_b_report.json", report)
    _write_json(root / "phase_b_lock.json", out)
    return out


# --- Phase C ----------------------------------------------------------------

def _pc_cfg(epsilon: float) -> dict[str, object]:
    return {
        "horizon": HORIZON, "variant": "para", "hidden": HIDDEN,
        "dropout": DROPOUT, "film_bound": 0.1, "epsilon": float(epsilon),
        "level_mu": 0.0, "level_sd": 1.0, "sigma_mu": 0.0, "sigma_sd": 1.0,
    }


def _merge_packs(train_pack, val_pack):
    return PcFraPack(
        np.concatenate([train_pack.starts, val_pack.starts]),
        np.concatenate([train_pack.embeddings, val_pack.embeddings]),
        np.concatenate([train_pack.partner_starts, val_pack.partner_starts]),
        HORIZON,
    )


def _calibrated_lookup(name, scale, q_grid, taus, bundle, starts, psrc):
    """Physical-unit calibrated Chronos trajectory keyed by window start."""
    lookup = {}
    if name.startswith("skew_b"):
        beta = float(name[len("skew_b"):])
        base_all = skew_readout(q_grid, taus, beta)
    else:
        base_all = readout(q_grid, taus, name)
    base_all = contract_prior(base_all, bundle, starts, SEQ_LEN, HORIZON)
    kind, value = scale
    for i, s in enumerate(starts):
        si = int(s)
        p0 = base_all[i]
        yp = psrc["pred_phys"][i]
        if kind == "scalar":
            p1 = scale_disagreement(p0[None], yp[None], float(value))[0]
        else:
            p1 = scale_disagreement_segments(
                p0[None], yp[None], tuple(float(g) for g in value)
            )[0]
        lookup[si] = p1.astype(np.float32)
    return lookup


def phase_c(config, records, root: Path) -> dict[str, object]:
    lock_b = json.loads((root / "phase_b_lock.json").read_text(encoding="utf-8"))
    taus = np.asarray(CHRONOS2_QUANTILES)
    table = {}
    for locked in lock_b["locked_priors"]:
        name = locked["candidate"]
        kind = locked["scale_rule_kind"]
        value = locked["scale_rule_value"]
        scale = (kind, value)
        tag = f"{name}__{locked['scale_rule_name']}"
        table[tag] = {}
        for station in DEV_STATIONS:
            station_dir = root / "phase_c" / tag.replace("+", "p") / station
            done = station_dir / "validation.json"
            if done.is_file():
                table[tag][station] = json.loads(
                    done.read_text(encoding="utf-8")
                )
                continue
            bundle = prev._load_bundle(station, config)
            params = dict(records[station]["params"])
            state = torch.load(
                prev._checkpoint(station), map_location="cpu",
                weights_only=True,
            )["state_dict"]
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            frozen, job = prev._build_frozen(
                station, config, bundle, params, state, device
            )
            batch = int(params.get("batch", 128))
            cache_path = prior_cache_path_v4(
                bundle.path, HORIZON, LOCKED_CONTEXT, LOCKED_NORM
            )
            cache_starts, q_grid, _ = load_grid_cache(cache_path)
            grid_lookup = {int(s): i for i, s in enumerate(cache_starts)}

            train_starts = bundle.train.starts  # L96: every window covered
            val_starts = bundle.validation.starts
            train_a = prev._infer(frozen, bundle, train_starts, batch, device)[1]
            val_a = prev._infer(frozen, bundle, val_starts, batch, device)[1]
            sd, mu = float(bundle.feature_sd[-1]), float(bundle.feature_mu[-1])

            def phys(arr):
                return arr["pred_std"] * sd + mu

            train_psrc = {"pred_phys": phys(train_a)}
            val_psrc = {"pred_phys": phys(val_a)}
            train_q = q_grid[[grid_lookup[int(s)] for s in train_starts]]
            val_q = q_grid[[grid_lookup[int(s)] for s in val_starts]]
            median_lookup = _calibrated_lookup(
                name, scale, train_q, taus, bundle, train_starts, train_psrc
            )
            median_lookup.update(_calibrated_lookup(
                name, scale, val_q, taus, bundle, val_starts, val_psrc
            ))
            psrc_lookup = {
                int(s): train_a["pred_std"][i]
                for i, s in enumerate(train_starts)
            }
            psrc_lookup.update({
                int(s): val_a["pred_std"][i]
                for i, s in enumerate(val_starts)
            })

            levels, sigmas = [], []
            for s in train_starts:
                lr, sg, _, _, _, _ = window_scalar_features(
                    bundle, int(s), SEQ_LEN, HORIZON
                )
                levels.append(lr)
                sigmas.append(sg)
            stats = {
                "level_mu": float(np.mean(levels)),
                "level_sd": float(np.std(levels) + 1e-8),
                "sigma_mu": float(np.mean(sigmas)),
                "sigma_sd": float(np.std(sigmas) + 1e-8),
            }
            epsilon_info = compute_epsilon(
                train_a["pred_std"] - train_a["target_std"],
                sd, bundle.capacity,
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
            run_params.update({
                "lr": ADAPTER_LR, "weight_decay": ADAPTER_WD,
                "scheduler": "none",
            })
            started = time.perf_counter()
            model, history, stopper, _ = _train_model(
                job, config, bundle, run_params,
                psrc_settings(params, config.psrc),
                epochs=EPOCHS, patience=PATIENCE,
                fm_embeddings=_merge_packs(train_pack, val_pack),
                initial_state_dict=state, train_pc_only=True,
                pc_fra_config=cfg,
            )
            _, arrays_b = _evaluate(
                model,
                _loader(bundle, val_starts, batch, False, SEED,
                        fm_embeddings=val_pack),
                bundle, device, collect=True,
            )
            station_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"state_dict": model.pc_fra_adapter.state_dict()},
                station_dir / "adapter_checkpoint.pt",
            )
            from formal.metrics import regression_metrics
            target_phys = arrays_b["target_std"] * sd + mu
            pred_b = np.maximum(0.0, arrays_b["pred_std"] * sd + mu)
            pred_a = np.maximum(0.0, val_a["pred_std"] * sd + mu)
            ma = regression_metrics(pred_a, target_phys, bundle.capacity)
            mb = regression_metrics(pred_b, target_phys, bundle.capacity)
            payload = {
                "station": station,
                "locked_prior": {
                    "context_rows": LOCKED_CONTEXT,
                    "input_norm": LOCKED_NORM,
                    "candidate": name,
                    "scale_rule_kind": kind,
                    "scale_rule_name": locked["scale_rule_name"],
                    "scale_rule_value": value,
                },
                "n_train_windows": int(len(train_starts)),
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
            _write_json(done, payload)
            table[tag][station] = payload

    decision = {"per_prior": {}}
    for tag, st in table.items():
        rmse_gain = {
            s: st[s]["A_frozen_psrc"]["rmse_physical"]
            - st[s]["B_para_locked_prior"]["rmse_physical"]
            for s in DEV_STATIONS
        }
        mae_gain = {
            s: st[s]["A_frozen_psrc"]["mae_physical"]
            - st[s]["B_para_locked_prior"]["mae_physical"]
            for s in DEV_STATIONS
        }
        promotes = all(rmse_gain[s] > 0 for s in DEV_STATIONS) and not all(
            mae_gain[s] < 0 for s in DEV_STATIONS
        )
        decision["per_prior"][tag] = {
            "rmse_gain_B_minus_A": rmse_gain,
            "mae_gain_B_minus_A": mae_gain,
            "B_promotes": bool(promotes),
            "per_station": {
                s: {
                    "A_rmse": st[s]["A_frozen_psrc"]["rmse_physical"],
                    "B_rmse": st[s]["B_para_locked_prior"]["rmse_physical"],
                    "A_mae": st[s]["A_frozen_psrc"]["mae_physical"],
                    "B_mae": st[s]["B_para_locked_prior"]["mae_physical"],
                    "best_epoch": st[s]["best_epoch"],
                }
                for s in DEV_STATIONS
            },
        }
    decision["any_prior_promotes"] = any(
        v["B_promotes"] for v in decision["per_prior"].values()
    )
    decision["test_evaluated"] = False
    _write_json(root / "phase_c" / "phase_c_decision.json", decision)
    return decision


# --- entry point ------------------------------------------------------------

def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/formal_experiment.json"))
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--protocol", action="store_true")
    parser.add_argument("--build-caches", action="store_true")
    parser.add_argument("--phase-a", action="store_true")
    parser.add_argument("--phase-b", action="store_true")
    parser.add_argument("--phase-c", action="store_true")
    args = parser.parse_args(argv)
    root = args.output_root

    if args.protocol:
        spec = protocol()
        print(json.dumps(spec, ensure_ascii=False, indent=2))
        print("phase A candidates:", list(PHASE_A_CANDIDATES))
        return 0

    config = load_experiment_config(args.config)
    records = load_best_parameters(config.best_parameters)
    root.mkdir(parents=True, exist_ok=True)

    if args.build_caches:
        build_caches(config, records, root)
        _write_status(root, "caches_built", stations=list(ALL_STATIONS),
                      context_rows=LOCKED_CONTEXT)
    if args.phase_a:
        d = phase_a(config, records, root)
        for item in d["selected"]:
            print(
                f"[phase-a] keep {item['candidate']:<20} "
                f"worst_rho={item['worst_station_residual_corr']:+.4f} "
                f"worst_dir={item['worst_station_direction_accuracy']:.4f} "
                f"worst_kappa_gap={item['worst_station_kappa_gap']:.3f}",
                flush=True,
            )
        _write_status(root, "phase_a_completed",
                      selected=[x["candidate"] for x in d["selected"]])
    if args.phase_b:
        out = phase_b(config, records, root)
        for lp in out["locked_priors"]:
            print(
                f"[phase-b] lock {lp['candidate']} + {lp['scale_rule_name']} "
                f"(all rho>0: {lp['all_stations_rho_positive']}, "
                f"worst |kappa-1|={lp['worst_station_kappa_gap']:.3f})",
                flush=True,
            )
        _write_status(root, "phase_b_completed",
                      locked=[
                          f"{x['candidate']}+{x['scale_rule_name']}"
                          for x in out["locked_priors"]
                      ])
    if args.phase_c:
        d = phase_c(config, records, root)
        for tag, v in d["per_prior"].items():
            print(f"[phase-c] {tag}: promotes={v['B_promotes']} "
                  f"rmse_gains={v['rmse_gain_B_minus_A']}", flush=True)
        _write_status(root, "phase_c_completed",
                      any_promotes=bool(d["any_prior_promotes"]),
                      test_evaluated=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
