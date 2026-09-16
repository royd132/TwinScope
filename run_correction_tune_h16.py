"""Correction-amplitude / training-stability tuning, L96-H16 (validation only).

The Chronos prior is LOCKED from chronos_l96_calib_v1 (context 96, raw
input, two surviving readouts each with disagreement scale gamma=0.50):

  * skew_b+0.50          P* = q50 + 0.5*(q90+q10-2*q50)
  * mean_q_full         trapezoid mean of the full 21-q grid

Nothing about Chronos, the adapter architecture, its position, physical
tokens, FiLM, PSRC or the information set X_{t-95:t} changes.

Step 0 (no tuning): train the baseline PARA adapter once per station per
locked prior with the registered hyperparameters, then evaluate the
analytic validation-only least-squares correction shrinkage

    alpha* = clip(sum_i Delta_i R_i / (sum_i Delta_i^2 + eps), 0, 1)

on the four comparison-subset stations (31, 1A, 02, HKUST). Gate: at
least one locked prior per station must have alpha*>0 and
RMSE(PSRC + alpha* Delta) < RMSE(PSRC) on validation.

Round 1 (per station, prior chosen on validation): 12 registered
lr x c x wd configurations with the correction bound parameterised as
epsilon = c * sigma_R (training PSRC residual std, standardized).

Round 2 (on the Round-1 winner): Huber beta in sigma_R units and
correction-energy / horizon-smoothness penalties (engine flags that
default to the exact previous behaviour).

Final control: same locked hyperparameters with the Chronos prior
permuted across windows (fixed seed) - RMSE_real must beat RMSE_shuffled.

Stages: --step0 / --round1 / --round2 / --shuffle-control.
Test windows are never read.
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from formal.chronos_prior import (
    CHRONOS2_QUANTILES,
    build_chronos2_grid_cache,
    load_grid_cache,
    prior_cache_path_v4,
)
from formal.config import (
    JobSpec,
    load_best_parameters,
    load_experiment_config,
)
from formal.engine import (
    _evaluate,
    _loader,
    _model_config,
    _train_model,
    build_model,
    psrc_settings,
)
from formal.metrics import regression_metrics
from formal.pc_fra import (
    PcFraPack,
    build_pack,
    compute_epsilon,
    stratified_partners,
    window_scalar_features,
)

import run_chronos_l96_calibration as calib
import run_chronos_prior_tuning_h16 as prev

SEQ_LEN = calib.SEQ_LEN
HORIZON = calib.HORIZON
SEED = calib.SEED
SHUFFLE_SEED = calib.SHUFFLE_SEED
HIDDEN = calib.HIDDEN
DROPOUT = calib.DROPOUT
EPOCHS = calib.EPOCHS
PATIENCE = calib.PATIENCE

STATIONS = calib.ALL_STATIONS  # 31, 02, 1A, HKUST
ROOT = Path("outputs/correction_tune_v1")
LOCKED_CONTEXT = 96
LOCKED_NORM = "raw"
LOCKED_PRIORS = (
    ("skew_b+0.50", ("scalar", 0.50)),
    ("mean_q_full", ("scalar", 0.50)),
)

# Registered Round 1 grid: lr x c x wd = 3 x 3 x 2 = 18 maximum; the
# campaign runs 12 balanced configurations per station.
ROUND1_LR = (3e-5, 1e-4, 3e-4)
ROUND1_C = (0.25, 0.50, 1.00)
ROUND1_WD = (0.0, 1e-3)
# 12 balanced triples (each level appears equally often).
ROUND1_GRID: tuple[tuple[float, float, float], ...] = (
    (3e-5, 0.25, 0.0),
    (3e-5, 0.50, 1e-3),
    (3e-5, 1.00, 0.0),
    (1e-4, 0.25, 1e-3),
    (1e-4, 0.25, 0.0),
    (1e-4, 0.50, 0.0),
    (1e-4, 0.50, 1e-3),
    (1e-4, 1.00, 1e-3),
    (3e-4, 0.25, 0.0),
    (3e-4, 0.50, 1e-3),
    (3e-4, 0.50, 0.0),
    (3e-4, 1.00, 0.0),
)

ROUND2_HUBER_BETA_C = (0.25, 0.50, 1.00)   # x sigma_R
ROUND2_LAMBDA_ENERGY = (0.0, 1e-3, 1e-2)
ROUND2_LAMBDA_SMOOTH = (0.0, 1e-3, 1e-2)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def _write_status(root: Path, stage: str, **details) -> None:
    path = root / "run_status.json"
    if path.is_file():
        status = json.loads(path.read_text(encoding="utf-8"))
    else:
        status = {"campaign": "correction_tune_v1", "history": []}
    status["current_stage"] = stage
    status.setdefault("history", []).append(
        {"stage": stage,
         "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         **details}
    )
    _write_json(path, status)


# --- generic adapter training ----------------------------------------------

def _merge_packs(train_pack, val_pack):
    return PcFraPack(
        np.concatenate([train_pack.starts, val_pack.starts]),
        np.concatenate([train_pack.embeddings, val_pack.embeddings]),
        np.concatenate([train_pack.partner_starts, val_pack.partner_starts]),
        HORIZON,
    )


def _station_cache(station, config):
    bundle = prev._load_bundle(station, config)
    path = prior_cache_path_v4(
        bundle.path, HORIZON, LOCKED_CONTEXT, LOCKED_NORM
    )
    starts, q_grid, _ = load_grid_cache(path)
    return bundle, {int(s): i for i, s in enumerate(starts)}, q_grid


def _test_grid_sidecar(root: Path, station, bundle, test_starts) -> Path:
    """Frozen-Chronos 21-q grid for test windows, separate sidecar cache."""
    out = root / "test_prior_cache" / (
        f"{station}__h{HORIZON}__ctx{LOCKED_CONTEXT}__{LOCKED_NORM}"
        "__test_grid.npz"
    )
    return build_chronos2_grid_cache(
        bundle, np.asarray(test_starts, dtype=np.int64),
        HORIZON, LOCKED_CONTEXT, LOCKED_NORM,
        seq_len=SEQ_LEN, out_path=out,
    )


def _station_cache_with_test(root: Path, station, config):
    bundle, dev_lookup, dev_grid = _station_cache(station, config)
    test_starts = np.asarray(bundle.test.starts, dtype=np.int64)
    missing = [int(s) for s in test_starts if int(s) not in dev_lookup]
    if missing:
        side = _test_grid_sidecar(root, station, bundle, test_starts)
        t_starts, t_grid, _ = load_grid_cache(side)
        if len(t_starts) != len(test_starts):
            raise ValueError(
                f"{station}: test sidecar dropped "
                f"{len(test_starts) - len(t_starts)} windows"
            )
        q_grid = np.concatenate([dev_grid, t_grid], axis=0)
        lookup = dict(dev_lookup)
        base = len(dev_grid)
        for i, s in enumerate(t_starts):
            lookup[int(s)] = base + i
    else:
        q_grid, lookup = dev_grid, dev_lookup
    return bundle, lookup, q_grid


def _shuffle_lookup(lookup, seed: int):
    """Permute prior trajectories across windows (marginal-preserving)."""
    keys = sorted(lookup)
    perm = np.random.default_rng(seed).permutation(len(keys))
    return {k: lookup[keys[perm[i]]] for i, k in enumerate(keys)}


def train_one(
    station, config, records, cand_name, scale, tag, out_root: Path,
    *, lr: float, weight_decay: float, epsilon: float,
    huber_beta_c: float = 1.0, lambda_energy: float = 0.0,
    lambda_smooth: float = 0.0, shuffle_prior: bool = False,
    skip_if_done: bool = True,
):
    """Train one PARA adapter; return validation arrays + metrics payload."""
    import dataclasses
    import time

    out_dir = out_root / tag / station
    done = out_dir / "validation.json"
    arrays_path = out_dir / "validation_arrays.npz"
    if skip_if_done and done.is_file() and arrays_path.is_file():
        payload = json.loads(done.read_text(encoding="utf-8"))
        arrays = dict(np.load(arrays_path))
        return payload, arrays

    bundle, grid_lookup, q_grid = _station_cache(station, config)
    params = dict(records[station]["params"])
    state = torch.load(
        prev._checkpoint(station), map_location="cpu", weights_only=True
    )["state_dict"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frozen, job = prev._build_frozen(station, config, bundle, params, state, device)
    batch = int(params.get("batch", 128))
    taus = np.asarray(CHRONOS2_QUANTILES)

    train_starts = bundle.train.starts
    val_starts = bundle.validation.starts
    train_a = prev._infer(frozen, bundle, train_starts, batch, device)[1]
    val_a = prev._infer(frozen, bundle, val_starts, batch, device)[1]
    sd, mu = float(bundle.feature_sd[-1]), float(bundle.feature_mu[-1])

    def phys(a):
        return a["pred_std"] * sd + mu

    train_q = q_grid[[grid_lookup[int(s)] for s in train_starts]]
    val_q = q_grid[[grid_lookup[int(s)] for s in val_starts]]
    median_lookup = calib._calibrated_lookup(
        cand_name, scale, train_q, taus, bundle, train_starts,
        {"pred_phys": phys(train_a)},
    )
    median_lookup.update(calib._calibrated_lookup(
        cand_name, scale, val_q, taus, bundle, val_starts,
        {"pred_phys": phys(val_a)},
    ))
    if shuffle_prior:
        train_lookup = _shuffle_lookup(
            {int(s): median_lookup[int(s)] for s in train_starts}, SEED,
        )
        val_lookup = _shuffle_lookup(
            {int(s): median_lookup[int(s)] for s in val_starts}, SEED + 1,
        )
        median_lookup = {**train_lookup, **val_lookup}
    psrc_lookup = {
        int(s): train_a["pred_std"][i] for i, s in enumerate(train_starts)
    }
    psrc_lookup.update({
        int(s): val_a["pred_std"][i] for i, s in enumerate(val_starts)
    })

    levels, sigmas = [], []
    for s in train_starts:
        lr_, sg, _, _, _, _ = window_scalar_features(
            bundle, int(s), SEQ_LEN, HORIZON
        )
        levels.append(lr_)
        sigmas.append(sg)
    stats = {
        "level_mu": float(np.mean(levels)),
        "level_sd": float(np.std(levels) + 1e-8),
        "sigma_mu": float(np.mean(sigmas)),
        "sigma_sd": float(np.std(sigmas) + 1e-8),
    }
    cfg = {
        "horizon": HORIZON, "variant": "para", "hidden": HIDDEN,
        "dropout": DROPOUT, "film_bound": 0.1, "epsilon": float(epsilon),
        "huber_beta": float(huber_beta_c),
        "energy_weight": float(lambda_energy),
        "smooth_weight": float(lambda_smooth),
        **{k: float(v) for k, v in stats.items()},
    }
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
    run_params.update({"lr": float(lr), "weight_decay": float(weight_decay),
                       "scheduler": "none"})
    started = __import__("time").perf_counter()
    model, history, stopper, _ = _train_model(
        job, config, bundle, run_params,
        psrc_settings(params, config.psrc),
        epochs=EPOCHS, patience=PATIENCE,
        fm_embeddings=_merge_packs(train_pack, val_pack),
        initial_state_dict=state, train_pc_only=True, pc_fra_config=cfg,
    )
    _, arrays_b = _evaluate(
        model,
        _loader(bundle, val_starts, batch, False, SEED, fm_embeddings=val_pack),
        bundle, device, collect=True,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.pc_fra_adapter.state_dict()},
               out_dir / "adapter_checkpoint.pt")

    target_phys = arrays_b["target_std"] * sd + mu
    pred_b = np.maximum(0.0, arrays_b["pred_std"] * sd + mu)
    pred_a = np.maximum(0.0, val_a["pred_std"] * sd + mu)
    arrays = {
        "psrc_std": val_a["pred_std"].astype(np.float32),
        "pred_b_std": arrays_b["pred_std"].astype(np.float32),
        "target_std": arrays_b["target_std"].astype(np.float32),
        "target_sd": np.float32(sd), "target_mu": np.float32(mu),
    }
    np.savez(arrays_path, **arrays)
    ma = regression_metrics(pred_a, target_phys, bundle.capacity)
    mb = regression_metrics(pred_b, target_phys, bundle.capacity)
    resid_std = float(np.std(
        train_a["pred_std"] - train_a["target_std"]
    ))
    payload = {
        "station": station, "tag": tag,
        "locked_prior": {"candidate": cand_name, "scale": list(scale)},
        "shuffled_prior": bool(shuffle_prior),
        "hyperparameters": {
            "lr": float(lr), "weight_decay": float(weight_decay),
            "epsilon": float(epsilon), "huber_beta_std": float(huber_beta_c),
            "lambda_energy": float(lambda_energy),
            "lambda_smooth": float(lambda_smooth),
            "hidden": HIDDEN, "dropout": DROPOUT,
        },
        "train_residual_std": resid_std,
        "best_epoch": stopper.best_epoch, "epochs_ran": len(history),
        "seconds": __import__("time").perf_counter() - started,
        "A_frozen_psrc": {
            "rmse_physical": float(ma["rmse_physical"]),
            "mae_physical": float(ma["mae_physical"]),
            "mbe_physical": float(ma["mbe_physical"]),
            "r2": float(ma["r2"]),
        },
        "B_adapter": {
            "rmse_physical": float(mb["rmse_physical"]),
            "mae_physical": float(mb["mae_physical"]),
            "mbe_physical": float(mb["mbe_physical"]),
            "r2": float(mb["r2"]),
        },
    }
    _write_json(done, payload)
    return payload, arrays


# --- Step 0: analytic shrinkage --------------------------------------------

def shrinkage_analysis(arrays: dict[str, np.ndarray]) -> dict[str, object]:
    """alpha* LS shrinkage of the adapter correction, validation only."""
    psrc = arrays["psrc_std"]
    pred = arrays["pred_b_std"]
    target = arrays["target_std"]
    delta = (pred - psrc).reshape(-1)
    resid = (target - psrc).reshape(-1)
    denom = float(np.sum(delta * delta))
    alpha_raw = float(np.sum(delta * resid) / denom) if denom > 0 else 0.0
    alpha = float(np.clip(alpha_raw, 0.0, 1.0))
    per_h = []
    for h in range(psrc.shape[1]):
        d = pred[:, h] - psrc[:, h]
        r = target[:, h] - psrc[:, h]
        den = float(np.sum(d * d))
        per_h.append(float(np.clip(np.sum(d * r) / den, 0.0, 1.0))
                    if den > 0 else 0.0)
    return {"alpha_star": alpha, "alpha_unclipped": alpha_raw,
            "alpha_per_horizon": per_h}


def _phys_metrics(psrc_std, pred_std, target_std, sd, mu, capacity):
    t = target_std * sd + mu
    p = np.maximum(0.0, pred_std * sd + mu)
    m = regression_metrics(p, t, capacity)
    return {k: float(m[k]) for k in ("rmse_physical", "mae_physical",
                                    "mbe_physical", "r2")}


def step0(config, records, root: Path) -> dict[str, object]:
    table = {}
    for station in STATIONS:
        bundle = prev._load_bundle(station, config)
        table[station] = {"capacity": float(bundle.capacity)}
        for cand, scale in LOCKED_PRIORS:
            # Baseline epsilon: registered p90 rule (identical to Phase C).
            # Need training PSRC residual arrays for the p90 statistic;
            # train_one recomputes them, but epsilon is a config input, so
            # derive it through a frozen inference pass first.
            params = dict(records[station]["params"])
            state = torch.load(
                prev._checkpoint(station), map_location="cpu",
                weights_only=True,
            )["state_dict"]
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            frozen, _ = prev._build_frozen(
                station, config, bundle, params, state, device
            )
            train_a = prev._infer(
                frozen, bundle, bundle.train.starts,
                int(params.get("batch", 128)), device,
            )[1]
            sd = float(bundle.feature_sd[-1])
            epsilon_info = compute_epsilon(
                train_a["pred_std"] - train_a["target_std"],
                sd, bundle.capacity,
            )
            tag = f"step0__{cand.replace('+','p')}"
            payload, arrays = train_one(
                station, config, records, cand, scale, tag, root,
                lr=calib.ADAPTER_LR, weight_decay=calib.ADAPTER_WD,
                epsilon=float(epsilon_info["epsilon"]),
            )
            shrink = shrinkage_analysis(arrays)
            psrc = arrays["psrc_std"]
            pred = arrays["pred_b_std"]
            tgt = arrays["target_std"]
            pred_alpha = psrc + shrink["alpha_star"] * (pred - psrc)
            m_a = _phys_metrics(psrc, psrc, tgt, float(arrays["target_sd"]),
                                float(arrays["target_mu"]), bundle.capacity)
            m_b = _phys_metrics(psrc, pred, tgt, float(arrays["target_sd"]),
                                float(arrays["target_mu"]), bundle.capacity)
            m_s = _phys_metrics(psrc, pred_alpha, tgt,
                                float(arrays["target_sd"]),
                                float(arrays["target_mu"]), bundle.capacity)
            table[station][cand] = {
                "alpha_star": shrink["alpha_star"],
                "alpha_unclipped": shrink["alpha_unclipped"],
                "alpha_per_horizon": shrink["alpha_per_horizon"],
                "epsilon_p90": float(epsilon_info["epsilon"]),
                "best_epoch": payload["best_epoch"],
                "A_psrc": m_a, "B_adapter": m_b,
                "B_shrunk": m_s,
                "shrink_helps_rmse": m_s["rmse_physical"] < m_a["rmse_physical"],
                "adapter_helps_rmse": m_b["rmse_physical"] < m_a["rmse_physical"],
            }
    gate = {}
    for station in STATIONS:
        best = None
        for cand, _ in LOCKED_PRIORS:
            row = table[station][cand]
            ok = row["alpha_star"] > 0.0 and row["shrink_helps_rmse"]
            if ok and (best is None
                       or row["B_shrunk"]["rmse_physical"]
                       < best[1]["B_shrunk"]["rmse_physical"]):
                best = (cand, row)
        gate[station] = {
            "passes": best is not None,
            "best_prior": best[0] if best else None,
        }
    decision = {
        "per_station": table,
        "gate": gate,
        "all_four_pass": all(v["passes"] for v in gate.values()),
        "formula": "alpha* = clip(sum Delta R / sum Delta^2, 0, 1)",
        "test_evaluated": False,
    }
    _write_json(root / "step0_report.json", decision)
    return decision


# --- Round 1: per-station lr x c(sigma_R bound) x wd grid ------------------

ROUND2_GRID: tuple[tuple[float, float, float], ...] = (
    # (huber_beta as multiple of sigma_R, lambda_energy, lambda_smooth)
    (1.00, 0.0, 0.0),       # Round-1 winner baseline (re-trained, same cfg)
    (0.25, 0.0, 0.0),
    (0.50, 0.0, 0.0),
    (1.00, 1e-3, 0.0),
    (1.00, 1e-2, 0.0),
    (1.00, 0.0, 1e-3),
    (1.00, 0.0, 1e-2),
    (0.50, 1e-2, 0.0),
    (0.50, 1e-2, 1e-2),
    (0.25, 1e-2, 1e-2),
)


def _step0_payload(root: Path, station: str, cand: str) -> dict:
    tag = f"step0__{cand.replace('+', 'p')}"
    return json.loads(
        (root / tag / station / "validation.json").read_text(encoding="utf-8")
    )


def _capacity(root: Path, station: str) -> float:
    rep = json.loads(
        (root / "step0_report.json").read_text(encoding="utf-8")
    )
    return float(rep["per_station"][station]["capacity"])


def _choose_prior(root: Path, station: str) -> str:
    """Station-level prior choice: lower baseline adapter validation RMSE."""
    cands = [c for c, _ in LOCKED_PRIORS]
    return min(
        cands,
        key=lambda c: _step0_payload(root, station, c)["B_adapter"][
            "rmse_physical"
        ],
    )


def _eligible(b_rmse: float, b_mae: float, a_rmse: float, a_mae: float) -> bool:
    return b_rmse < a_rmse and b_mae <= 1.005 * a_mae


def _shrink_trial(arrays: dict, station: str, records_payload_cap,
                  a_rmse: float, a_mae: float) -> dict:
    """Analytic alpha* shrinkage for one trained adapter (validation only).

    The deployed FM prediction is yP + alpha* * Delta with alpha* solved by
    LS on the validation windows (user's prescribed correction shrinkage).
    """
    psrc = arrays["psrc_std"]
    pred = arrays["pred_b_std"]
    tgt = arrays["target_std"]
    sd = float(arrays["target_sd"])
    mu = float(arrays["target_mu"])
    sh = shrinkage_analysis(arrays)
    pred_alpha = psrc + sh["alpha_star"] * (pred - psrc)
    m_s = _phys_metrics(psrc, pred_alpha, tgt, sd, mu, records_payload_cap)
    return {
        "alpha_star": sh["alpha_star"],
        "alpha_unclipped": sh["alpha_unclipped"],
        "shrunk_rmse": m_s["rmse_physical"],
        "shrunk_mae": m_s["mae_physical"],
        "shrunk_mbe": m_s["mbe_physical"],
        "eligible_shrunk": _eligible(
            m_s["rmse_physical"], m_s["mae_physical"], a_rmse, a_mae
        ),
    }


def round1(config, records, root: Path) -> dict[str, object]:
    chosen = {}
    trials = {}
    for station in STATIONS:
        cand = _choose_prior(root, station)
        scale = dict(LOCKED_PRIORS)[cand]
        chosen[station] = cand
        base = _step0_payload(root, station, cand)
        sigma_r = float(base["train_residual_std"])
        a_rmse = float(base["A_frozen_psrc"]["rmse_physical"])
        a_mae = float(base["A_frozen_psrc"]["mae_physical"])
        rows = []
        for idx, (lr, c, wd) in enumerate(ROUND1_GRID):
            tag = f"round1__{cand.replace('+','p')}__t{idx:02d}"
            payload, arrays = train_one(
                station, config, records, cand, scale, tag, root,
                lr=float(lr), weight_decay=float(wd),
                epsilon=float(c * sigma_r),
            )
            shrink = _shrink_trial(
                arrays, station, _capacity(root, station), a_rmse, a_mae,
            )
            rows.append({
                "trial": idx, "lr": float(lr), "c": float(c),
                "weight_decay": float(wd), "epsilon": float(c * sigma_r),
                "best_epoch": payload["best_epoch"],
                "rmse": payload["B_adapter"]["rmse_physical"],
                "mae": payload["B_adapter"]["mae_physical"],
                "mbe": payload["B_adapter"]["mbe_physical"],
                "eligible": _eligible(
                    payload["B_adapter"]["rmse_physical"],
                    payload["B_adapter"]["mae_physical"], a_rmse, a_mae,
                ),
                **shrink,
            })
        feasible = [r for r in rows if r["eligible"]]
        feasible_s = [r for r in rows if r["eligible_shrunk"]]
        winner = min(feasible or rows, key=lambda r: r["rmse"])
        winner_s = min(feasible_s or rows, key=lambda r: r["shrunk_rmse"])
        trials[station] = {
            "chosen_prior": cand, "train_residual_std": sigma_r,
            "A_rmse": a_rmse, "A_mae": a_mae,
            "trials": rows,
            "winner": winner,
            "winner_shrunk": winner_s,
            "station_improves": bool(feasible_s),
            "station_improves_raw": bool(feasible),
        }
    decision = {
        "per_station": trials,
        "chosen_priors": chosen,
        "n_improving_stations": sum(
            1 for v in trials.values() if v["station_improves"]
        ),
        "constraints": ("RMSE(yP+a*D) < RMSE_PSRC and "
                        "MAE(yP+a*D) <= 1.005 MAE_PSRC; "
                        "alpha* analytic on validation"),
        "test_evaluated": False,
    }
    _write_json(root / "round1_report.json", decision)
    return decision


# --- Round 2: Huber beta + energy/smoothness on the Round-1 winner ---------

def round2(config, records, root: Path) -> dict[str, object]:
    r1 = json.loads((root / "round1_report.json").read_text(encoding="utf-8"))
    out = {}
    for station in STATIONS:
        body = r1["per_station"][station]
        cand = body["chosen_prior"]
        scale = dict(LOCKED_PRIORS)[cand]
        w = body.get("winner_shrunk", body["winner"])
        sigma_r = float(body["train_residual_std"])
        a_rmse, a_mae = float(body["A_rmse"]), float(body["A_mae"])
        cap = _capacity(root, station)
        rows = []
        for idx, (beta_c, lam_e, lam_s) in enumerate(ROUND2_GRID):
            tag = f"round2__{cand.replace('+','p')}__t{idx:02d}"
            payload, arrays = train_one(
                station, config, records, cand, scale, tag, root,
                lr=float(w["lr"]), weight_decay=float(w["weight_decay"]),
                epsilon=float(w["c"] * sigma_r),
                huber_beta_c=float(beta_c * sigma_r),
                lambda_energy=float(lam_e),
                lambda_smooth=float(lam_s),
            )
            shrink = _shrink_trial(arrays, station, cap, a_rmse, a_mae)
            rows.append({
                "trial": idx, "huber_beta_c": float(beta_c),
                "huber_beta_std": float(beta_c * sigma_r),
                "lambda_energy": float(lam_e),
                "lambda_smooth": float(lam_s),
                "best_epoch": payload["best_epoch"],
                "rmse": payload["B_adapter"]["rmse_physical"],
                "mae": payload["B_adapter"]["mae_physical"],
                "mbe": payload["B_adapter"]["mbe_physical"],
                "eligible": _eligible(
                    payload["B_adapter"]["rmse_physical"],
                    payload["B_adapter"]["mae_physical"], a_rmse, a_mae,
                ),
                **shrink,
            })
        feasible = [r for r in rows if r["eligible_shrunk"]]
        winner = min(feasible or rows, key=lambda r: r["shrunk_rmse"])
        out[station] = {
            "chosen_prior": cand,
            "round1_hyperparameters": {
                "lr": w["lr"], "c": w["c"], "weight_decay": w["weight_decay"],
                "epsilon": w["epsilon"],
            },
            "A_rmse": a_rmse, "A_mae": a_mae,
            "trials": rows, "winner": winner,
            "station_improves": bool(feasible),
        }
    decision = {
        "per_station": out,
        "n_improving_stations": sum(
            1 for v in out.values() if v["station_improves"]
        ),
        "test_evaluated": False,
    }
    _write_json(root / "round2_report.json", decision)
    return decision


# --- Shuffled-prior control on the locked per-station configuration --------

def _locked_config(root: Path, station: str) -> dict:
    """Final lock = best of Round-1 winner (beta=1.0 std) and Round-2 grid.

    Round 1 trained with Huber beta fixed at 1.0 standardized unit, while
    Round 2 searches beta in {.25,.5,1.0} sigma_R; the R1 point is kept as
    an explicit fallback candidate.
    """
    r1 = json.loads((root / "round1_report.json").read_text(encoding="utf-8"))
    r2 = json.loads((root / "round2_report.json").read_text(encoding="utf-8"))
    b1 = r1["per_station"][station]["winner_shrunk"]
    b2 = r2["per_station"][station]["winner"]
    cand_r1 = {
        "source": "round1", "lr": float(b1["lr"]), "c": float(b1["c"]),
        "weight_decay": float(b1["weight_decay"]),
        "epsilon": float(b1["epsilon"]),
        "huber_beta_std": 1.0,
        "lambda_energy": 0.0, "lambda_smooth": 0.0,
        "alpha_star": float(b1["alpha_star"]),
        "shrunk_rmse": float(b1["shrunk_rmse"]),
        "shrunk_mae": float(b1["shrunk_mae"]),
        "raw_rmse": float(b1["rmse"]), "raw_mae": float(b1["mae"]),
        "trial_tag": f"round1__t{b1['trial']:02d}",
    }
    cand_r2 = {
        "source": "round2",
        "lr": float(r2["per_station"][station]["round1_hyperparameters"]["lr"]),
        "c": float(r2["per_station"][station]["round1_hyperparameters"]["c"]),
        "weight_decay": float(
            r2["per_station"][station]["round1_hyperparameters"]["weight_decay"]
        ),
        "epsilon": float(
            r2["per_station"][station]["round1_hyperparameters"]["epsilon"]
        ),
        "huber_beta_std": float(b2["huber_beta_std"]),
        "lambda_energy": float(b2["lambda_energy"]),
        "lambda_smooth": float(b2["lambda_smooth"]),
        "alpha_star": float(b2["alpha_star"]),
        "shrunk_rmse": float(b2["shrunk_rmse"]),
        "shrunk_mae": float(b2["shrunk_mae"]),
        "raw_rmse": float(b2["rmse"]), "raw_mae": float(b2["mae"]),
        "trial_tag": f"round2__t{b2['trial']:02d}",
    }
    return min((cand_r1, cand_r2), key=lambda x: x["shrunk_rmse"])


def shuffle_control(config, records, root: Path) -> dict[str, object]:
    r2 = json.loads((root / "round2_report.json").read_text(encoding="utf-8"))
    out = {}
    for station in STATIONS:
        cand = r2["per_station"][station]["chosen_prior"]
        scale = dict(LOCKED_PRIORS)[cand]
        lock = _locked_config(root, station)
        tag = f"shuffle__{cand.replace('+','p')}__{lock['source']}"
        payload, arrays = train_one(
            station, config, records, cand, scale, tag, root,
            lr=lock["lr"], weight_decay=lock["weight_decay"],
            epsilon=lock["epsilon"],
            huber_beta_c=lock["huber_beta_std"],
            lambda_energy=lock["lambda_energy"],
            lambda_smooth=lock["lambda_smooth"],
            shuffle_prior=True,
        )
        cap = _capacity(root, station)
        sh_shuf = _shrink_trial(
            arrays, station, cap,
            float(r2["per_station"][station]["A_rmse"]),
            float(r2["per_station"][station]["A_mae"]),
        )
        real_rmse = float(lock["shrunk_rmse"])
        shuf_rmse = float(sh_shuf["shrunk_rmse"])
        out[station] = {
            "chosen_prior": cand,
            "locked_source": lock["source"],
            "locked_hyperparameters": {
                k: lock[k] for k in (
                    "lr", "c", "weight_decay", "epsilon", "huber_beta_std",
                    "lambda_energy", "lambda_smooth",
                )
            },
            "rmse_real": real_rmse,
            "mae_real": float(lock["shrunk_mae"]),
            "alpha_real": float(lock["alpha_star"]),
            "rmse_shuffled": shuf_rmse,
            "mae_shuffled": float(sh_shuf["shrunk_mae"]),
            "alpha_shuffled": float(sh_shuf["alpha_star"]),
            "rmse_shuffled_raw": float(payload["B_adapter"]["rmse_physical"]),
            "real_beats_shuffled": real_rmse < shuf_rmse,
            "rmse_psrc": float(r2["per_station"][station]["A_rmse"]),
            "mae_psrc": float(r2["per_station"][station]["A_mae"]),
        }
    decision = {
        "per_station": out,
        "real_beats_shuffled_all": all(
            v["real_beats_shuffled"] for v in out.values()
        ),
        "test_evaluated": False,
    }
    _write_json(root / "shuffle_control_report.json", decision)
    return decision


# --- One-time test confirmation with locked artefacts -----------------------

FORMAL_PSRC_TEST_RMSE = {
    "dkasc_site31": 0.291144,
    "dkasc_site1a": 0.738819,
    "pvod_station02": 1.675506,
    "hkust": 1.843531,
}


def _adapter_dir(root: Path, cand: str, lock: dict) -> Path:
    cand_p = cand.replace("+", "p")
    stage, trial = lock["trial_tag"].split("__")
    return root / f"{stage}__{cand_p}__{trial}"


def _rebuild_model(station, config, records, bundle, state, lock, device):
    """Reconstruct the exact locked model and load the saved adapter."""
    params = dict(records[station]["params"])
    job = JobSpec(station, "psrc", SEQ_LEN, HORIZON, SEED)
    levels, sigmas = [], []
    for s in bundle.train.starts:
        lr_, sg, _, _, _, _ = window_scalar_features(
            bundle, int(s), SEQ_LEN, HORIZON
        )
        levels.append(lr_)
        sigmas.append(sg)
    cfg = {
        "horizon": HORIZON, "variant": "para", "hidden": HIDDEN,
        "dropout": DROPOUT, "film_bound": 0.1,
        "epsilon": float(lock["epsilon"]),
        "huber_beta": float(lock["huber_beta_std"]),
        "energy_weight": float(lock["lambda_energy"]),
        "smooth_weight": float(lock["lambda_smooth"]),
        "level_mu": float(np.mean(levels)),
        "level_sd": float(np.std(levels) + 1e-8),
        "sigma_mu": float(np.mean(sigmas)),
        "sigma_sd": float(np.std(sigmas) + 1e-8),
    }
    model = build_model(
        job.model,
        _model_config(job, bundle, params, True, pc_fra=cfg),
    ).to(device)
    incompatible = model.load_state_dict(state, strict=False)
    missing = [
        n for n in incompatible.missing_keys
        if not n.startswith("pc_fra_adapter.")
    ]
    if incompatible.unexpected_keys or missing:
        raise ValueError(f"frozen warm-start mismatch: {missing}")
    return model, cfg


def test_confirm(config, records, root: Path) -> dict[str, object]:
    """Evaluate PSRC / FM / shuffled-FM on test ONCE, hyperparameters frozen.

    alpha* is the value locked on validation; it is never recomputed on
    test. The locked adapters are loaded from disk (no retraining).
    """
    shuf_rep = json.loads(
        (root / "shuffle_control_report.json").read_text(encoding="utf-8")
    )
    out = {}
    for station in STATIONS:
        body = shuf_rep["per_station"][station]
        cand = body["chosen_prior"]
        scale = dict(LOCKED_PRIORS)[cand]
        lock = _locked_config(root, station)
        cap = _capacity(root, station)

        bundle, grid_lookup, q_grid = _station_cache_with_test(
            root, station, config
        )
        test_starts = np.asarray(bundle.test.starts)
        if any(int(s) not in grid_lookup for s in test_starts):
            raise ValueError(f"{station}: test windows still missing after "
                             "sidecar build")
        params = dict(records[station]["params"])
        state = torch.load(
            prev._checkpoint(station), map_location="cpu", weights_only=True,
        )["state_dict"]
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        frozen, _ = prev._build_frozen(
            station, config, bundle, params, state, device
        )
        batch = int(params.get("batch", 128))
        taus = np.asarray(CHRONOS2_QUANTILES)
        test_a = prev._infer(frozen, bundle, test_starts, batch, device)[1]
        sd = float(bundle.feature_sd[-1])
        mu = float(bundle.feature_mu[-1])

        def phys(a):
            return a["pred_std"] * sd + mu

        test_q = q_grid[[grid_lookup[int(s)] for s in test_starts]]
        prior_lookup = calib._calibrated_lookup(
            cand, scale, test_q, taus, bundle, test_starts,
            {"pred_phys": phys(test_a)},
        )
        psrc_lookup = {
            int(s): test_a["pred_std"][i]
            for i, s in enumerate(test_starts)
        }
        partners, _ = stratified_partners(
            test_starts, bundle, SEQ_LEN, HORIZON, SHUFFLE_SEED + 2
        )
        test_pack = build_pack(
            bundle, test_starts, SEQ_LEN, HORIZON,
            prior_lookup, psrc_lookup, partners,
        )
        shuf_lookup = _shuffle_lookup(dict(prior_lookup), SEED + 2)
        shuf_pack = build_pack(
            bundle, test_starts, SEQ_LEN, HORIZON,
            shuf_lookup, psrc_lookup, partners,
        )

        def eval_adapter(adapter_path: Path, pack: PcFraPack):
            model, _cfg = _rebuild_model(
                station, config, records, bundle, state, lock, device
            )
            model.pc_fra_adapter.load_state_dict(
                torch.load(adapter_path, map_location=device,
                           weights_only=True)["state_dict"],
                strict=True,
            )
            model.eval()
            return _evaluate(
                model,
                _loader(bundle, test_starts, batch, False, SEED,
                        fm_embeddings=pack),
                bundle, device, collect=True,
            )[1]

        real_dir = _adapter_dir(root, cand, lock)
        real_ckpt = real_dir / station / "adapter_checkpoint.pt"
        shuf_dir = root / (
            f"shuffle__{cand.replace('+','p')}__{lock['source']}"
        )
        shuf_ckpt = shuf_dir / station / "adapter_checkpoint.pt"
        if not shuf_ckpt.is_file():
            raise FileNotFoundError(f"missing shuffled adapter: {shuf_ckpt}")
        arrays_real = eval_adapter(real_ckpt, test_pack)
        arrays_shuf = eval_adapter(shuf_ckpt, shuf_pack)

        psrc_s = test_a["pred_std"]
        tgt_s = test_a["target_std"]

        def with_alpha(arrays, alpha):
            pred_s = psrc_s + alpha * (arrays["pred_std"] - psrc_s)
            m = _phys_metrics(psrc_s, pred_s, tgt_s, sd, mu, cap)
            return m

        m_a = _phys_metrics(psrc_s, psrc_s, tgt_s, sd, mu, cap)
        m_real = with_alpha(arrays_real, float(lock["alpha_star"]))
        m_real_raw = _phys_metrics(
            psrc_s, arrays_real["pred_std"], tgt_s, sd, mu, cap
        )
        m_shuf = with_alpha(arrays_shuf, float(body["alpha_shuffled"]))

        station_out = {
            "chosen_prior": cand,
            "locked_source": lock["source"],
            "locked_hyperparameters": {
                k: lock[k] for k in (
                    "lr", "c", "weight_decay", "epsilon", "huber_beta_std",
                    "lambda_energy", "lambda_smooth",
                )
            },
            "alpha_from_validation": float(lock["alpha_star"]),
            "n_test_windows": int(len(test_starts)),
            "A_psrc": m_a,
            "FM_real": m_real,
            "FM_real_raw_no_shrink": m_real_raw,
            "FM_shuffled": m_shuf,
            "formal_table_psrc_rmse": FORMAL_PSRC_TEST_RMSE[station],
            "psrc_matches_formal_table": abs(
                m_a["rmse_physical"] - FORMAL_PSRC_TEST_RMSE[station]
            ) < 5e-6,
            "real_beats_psrc": (
                m_real["rmse_physical"] < m_a["rmse_physical"]
                and m_real["mae_physical"] <= 1.005 * m_a["mae_physical"]
            ),
            "real_beats_shuffled": (
                m_real["rmse_physical"] < m_shuf["rmse_physical"]
            ),
        }
        out[station] = station_out
        tdir = root / "test_confirm" / station
        tdir.mkdir(parents=True, exist_ok=True)
        np.savez(
            tdir / "test_arrays.npz",
            psrc_std=psrc_s.astype(np.float32),
            pred_real_std=arrays_real["pred_std"].astype(np.float32),
            pred_shuffled_std=arrays_shuf["pred_std"].astype(np.float32),
            target_std=tgt_s.astype(np.float32),
            target_sd=np.float32(sd), target_mu=np.float32(mu),
        )
    decision = {
        "per_station": out,
        "all_four_beat_psrc": all(v["real_beats_psrc"] for v in out.values()),
        "all_four_beat_shuffled": all(
            v["real_beats_shuffled"] for v in out.values()
        ),
        "all_psrc_match_formal_table": all(
            v["psrc_matches_formal_table"] for v in out.values()
        ),
        "protocol": ("single test unblinding; adapters loaded from locked "
                     "development artefacts; alpha* fixed from validation"),
        "test_read_once": True,
    }
    _write_json(root / "test_confirmation_report.json", decision)
    return decision


# --- entry point ------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/formal_experiment.json"))
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--step0", action="store_true")
    parser.add_argument("--round1", action="store_true")
    parser.add_argument("--round2", action="store_true")
    parser.add_argument("--shuffle-control", action="store_true")
    parser.add_argument("--test-confirm", action="store_true",
                        help="one-time locked test unblinding")
    args = parser.parse_args(argv)
    root = args.output_root
    config = load_experiment_config(args.config)
    records = load_best_parameters(config.best_parameters)
    root.mkdir(parents=True, exist_ok=True)

    if args.step0:
        d = step0(config, records, root)
        for station, g in d["gate"].items():
            row = d["per_station"][station]
            line = f"[step0] {station:<16} pass={g['passes']} best={g['best_prior']}"
            for cand, _ in LOCKED_PRIORS:
                r = row[cand]
                line += (f" | {cand}: a*={r['alpha_star']:.3f} "
                         f"RMSE {r['A_psrc']['rmse_physical']:.4f}->"
                         f"{r['B_adapter']['rmse_physical']:.4f}->"
                         f"{r['B_shrunk']['rmse_physical']:.4f}")
            print(line, flush=True)
        print("[step0] ALL FOUR PASS:", d["all_four_pass"], flush=True)
        _write_status(root, "step0_completed",
                      all_four_pass=bool(d["all_four_pass"]))
    if args.round1:
        d = round1(config, records, root)
        for station, v in d["per_station"].items():
            w = v["winner_shrunk"]
            print(
                f"[round1] {station:<16} prior={v['chosen_prior']} "
                f"lr={w['lr']:.0e} c={w['c']} wd={w['weight_decay']:.0e} "
                f"a*={w['alpha_star']:.3f} RMSE={v['A_rmse']:.5f}->"
                f"{w['shrunk_rmse']:.5f} improves={v['station_improves']}",
                flush=True,
            )
        _write_status(root, "round1_completed",
                      n_improving=d["n_improving_stations"])
    if args.round2:
        d = round2(config, records, root)
        for station, v in d["per_station"].items():
            w = v["winner"]
            print(
                f"[round2] {station:<16} beta_c={w['huber_beta_c']} "
                f"lE={w['lambda_energy']:.0e} lS={w['lambda_smooth']:.0e} "
                f"a*={w['alpha_star']:.3f} RMSE={v['A_rmse']:.5f}->"
                f"{w['shrunk_rmse']:.5f} improves={v['station_improves']}",
                flush=True,
            )
        _write_status(root, "round2_completed",
                      n_improving=d["n_improving_stations"])
    if args.shuffle_control:
        d = shuffle_control(config, records, root)
        for station, v in d["per_station"].items():
            print(
                f"[shuffle] {station:<16} lock={v['locked_source']} "
                f"real={v['rmse_real']:.5f} (a={v['alpha_real']:.2f}) "
                f"shuffled={v['rmse_shuffled']:.5f} "
                f"(a={v['alpha_shuffled']:.2f}) "
                f"real_wins={v['real_beats_shuffled']}", flush=True,
            )
        _write_status(root, "shuffle_completed",
                      real_beats_shuffled_all=bool(
                          d["real_beats_shuffled_all"]
                      ))
    if args.test_confirm:
        d = test_confirm(config, records, root)
        for station, v in d["per_station"].items():
            a, f, s = v["A_psrc"], v["FM_real"], v["FM_shuffled"]
            gain = (a["rmse_physical"] - f["rmse_physical"]) \
                / a["rmse_physical"] * 100
            print(
                f"[test] {station:<16} n={v['n_test_windows']} "
                f"PSRC={a['rmse_physical']:.5f} FM={f['rmse_physical']:.5f} "
                f"({gain:+.2f}%) shuf={s['rmse_physical']:.5f} "
                f"beatPSRC={v['real_beats_psrc']} "
                f"beatSHUF={v['real_beats_shuffled']} "
                f"psrc_match={v['psrc_matches_formal_table']}", flush=True,
            )
        print("[test] ALL BEAT PSRC:", d["all_four_beat_psrc"], flush=True)
        print("[test] ALL BEAT SHUFFLED:", d["all_four_beat_shuffled"],
              flush=True)
        print("[test] PSRC MATCHES FORMAL TABLE:",
              d["all_psrc_match_formal_table"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
