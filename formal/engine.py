"""Training and evaluation engine for the locked formal protocol."""

from __future__ import annotations

import hashlib
import random
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)
from torch.utils.data import DataLoader, Dataset

from formal.config import (
    DETERMINISTIC_MODELS,
    PUBLIC_BASELINE_MODELS,
    ExperimentConfig,
    JobSpec,
    load_best_parameters,
)
from formal.data import DatasetBundle, load_dataset
from formal.metrics import regression_metrics
from layers.physical_semantic import (
    physical_token_statistics,
    resolve_physical_roles,
)
from models.factory import build_model


def psrc_settings(
    params: dict[str, object], defaults: dict[str, object]
) -> dict[str, object]:
    settings = dict(defaults)
    for key in (
        "semantic_loss_weight",
        "residual_alpha",
        "base_anchor_weight",
        "gate",
        "max_gate",
        "max_correction",
    ):
        if key in params:
            settings[key] = params[key]
    settings.setdefault("max_gate", 1.0)
    settings.setdefault("max_correction", 0.5)
    return settings


class EarlyStopping:
    def __init__(self, patience: int) -> None:
        if patience < 1:
            raise ValueError("patience must be positive")
        self.patience = int(patience)
        self.best_score = float("inf")
        self.best_epoch = 0
        self.bad_epochs = 0
        self.epoch = 0

    def update(self, score: float) -> bool:
        self.epoch += 1
        if score < self.best_score:
            self.best_score = float(score)
            self.best_epoch = self.epoch
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def psrc_residual_loss(
    base: torch.Tensor,
    correction: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Train correction toward a detached base residual."""
    residual = target - base.detach()
    mask = torch.ones_like(target) if mask is None else mask.to(target.dtype)
    per_window = (residual.abs() * mask).sum(1) / mask.sum(1).clamp_min(1.0)
    weights = 1.0 + float(alpha) * per_window / per_window.mean().clamp_min(1e-8)
    weighted_mask = mask * weights[:, None]
    return ((correction - residual).abs() * weighted_mask).sum() / (
        weighted_mask.sum().clamp_min(1.0)
    )


def point_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    bundle: DatasetBundle,
    kind: str,
    huber_delta: float,
    daylight_weight: float,
    ramp_weight: float,
) -> torch.Tensor:
    weights = torch.ones_like(target)
    if daylight_weight > 0.0:
        physical_target = target * bundle.feature_sd[-1] + bundle.feature_mu[-1]
        daylight = (physical_target > 0.01 * bundle.capacity).to(target.dtype)
        weights = 1.0 + float(daylight_weight) * daylight
        weights = weights / weights.mean().clamp_min(1e-8)
    if kind == "huber":
        point = F.huber_loss(
            prediction, target, reduction="none", delta=float(huber_delta)
        )
    elif kind == "mse":
        point = (prediction - target).square()
    else:
        raise ValueError(f"unknown point loss: {kind}")
    loss = (point * weights).mean()
    if ramp_weight > 0.0 and target.shape[1] > 1:
        pred_physical = prediction * bundle.feature_sd[-1] + bundle.feature_mu[-1]
        target_physical = target * bundle.feature_sd[-1] + bundle.feature_mu[-1]
        ramp_error = (
            torch.diff(pred_physical, dim=1) - torch.diff(target_physical, dim=1)
        ).abs()
        loss = loss + float(ramp_weight) * ramp_error.mean() / (
            bundle.capacity * bundle.sample_hours
        )
    return loss


class ForecastDataset(Dataset):
    def __init__(
        self,
        bundle: DatasetBundle,
        starts: np.ndarray,
        fm_embeddings: object | None = None,
    ) -> None:
        self.bundle = bundle
        self.starts = np.asarray(starts, dtype=np.int64)
        self.seq_len = bundle.train.seq_len
        self.horizon = bundle.train.horizon
        # Optional precomputed frozen foundation-model window embeddings,
        # indexed by the window's absolute start row. ``None`` (the formal
        # path) keeps the historical 4-tuple contract used by every model.
        self.fm_embeddings = fm_embeddings

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int):
        start = int(self.starts[index])
        target_start = start + self.seq_len
        history = self.bundle.values[start:target_start]
        target = self.bundle.values[target_start : target_start + self.horizon, -1]
        future_solar = self.bundle.values[
            target_start : target_start + self.horizon
        ][:, list(self.bundle.solar_future_indices)]
        timestamp = self.bundle.timestamps[target_start].astype("datetime64[m]")
        minute = int(timestamp.astype(np.int64) % (24 * 60))
        cycle = int(round(minute / (24 * 60) * 96)) % 96
        if self.fm_embeddings is None:
            return (
                torch.from_numpy(history),
                torch.from_numpy(target),
                torch.tensor(cycle, dtype=torch.long),
                torch.from_numpy(np.ascontiguousarray(future_solar)),
            )
        pack_vector = torch.from_numpy(
            self.fm_embeddings[start].astype(np.float32)
        )
        if getattr(self.fm_embeddings, "kind", None) == "pc_fra":
            # Sixth element: the stratified-shuffle partner's history,
            # used only by the physical-token shuffle diagnostic to
            # recompute that window's six frozen physical tokens.
            partner_start = int(self.fm_embeddings.partner_start(start))
            partner_history = self.bundle.values[
                partner_start : partner_start + self.seq_len
            ]
            return (
                torch.from_numpy(history),
                torch.from_numpy(target),
                torch.tensor(cycle, dtype=torch.long),
                torch.from_numpy(np.ascontiguousarray(future_solar)),
                pack_vector,
                torch.from_numpy(np.ascontiguousarray(partner_history)),
            )
        return (
            torch.from_numpy(history),
            torch.from_numpy(target),
            torch.tensor(cycle, dtype=torch.long),
            torch.from_numpy(np.ascontiguousarray(future_solar)),
            pack_vector,
        )


def _forward_batch(model, batch, device: torch.device) -> torch.Tensor:
    """Call a model on a 4-tuple (formal), 5-tuple (FM cache) or 6-tuple."""
    if len(batch) == 4:
        history, target, cycle, future_solar = batch
        return model(
            history.to(device),
            cycle.to(device),
            future_solar.to(device),
            None,
            None,
        ).squeeze(-1)
    if len(batch) not in {5, 6}:
        raise ValueError(f"unexpected batch tuple length: {len(batch)}")
    history, target, cycle, future_solar, fm_context = batch[:5]
    if not getattr(model, "accepts_fm_context", False):
        raise ValueError("FM embeddings were loaded but the model cannot consume them")
    pc_partner = batch[5].to(device) if len(batch) == 6 else None
    return model(
        history.to(device),
        cycle.to(device),
        future_solar.to(device),
        None,
        None,
        fm_context=fm_context.to(device),
        pc_partner=pc_partner,
    ).squeeze(-1)


def _evenly_sample(starts: np.ndarray, limit: int) -> np.ndarray:
    if len(starts) <= limit:
        return starts
    positions = np.linspace(0, len(starts) - 1, limit, dtype=np.int64)
    return starts[positions]


WARMUP_EPOCHS = 5


def _make_scheduler(params: dict[str, object], optimizer, epochs: int):
    """Optimization-only schedule policy; model architecture never changes here."""
    kind = params.get("scheduler")
    if kind == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs)
    if kind == "cosine_warmup":
        if epochs <= WARMUP_EPOCHS:
            raise ValueError("cosine_warmup needs more epochs than the warm-up length")
        warmup = LinearLR(
            optimizer, start_factor=0.2, total_iters=WARMUP_EPOCHS
        )
        cosine = CosineAnnealingLR(
            optimizer, T_max=epochs - WARMUP_EPOCHS
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[WARMUP_EPOCHS],
        )
    if kind in (None, "none"):
        return None
    raise ValueError(f"unsupported scheduler: {kind}")


def _training_starts(
    bundle: DatasetBundle,
    params: dict[str, object],
    limit: int,
    seed: int,
    epoch: int,
) -> np.ndarray:
    starts = bundle.train.starts
    policy = str(params.get("sampling_policy", "uniform"))
    if len(starts) <= limit:
        return starts
    if policy == "uniform":
        return _evenly_sample(starts, limit)
    if policy == "epoch_uniform":
        rng = np.random.default_rng(seed + epoch)
        return np.sort(rng.choice(starts, size=limit, replace=False))
    raise ValueError(f"unsupported formal sampling policy: {policy}")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _solar_geometry(bundle: DatasetBundle) -> dict[str, object] | None:
    """Audited deterministic future geometry consumed by smart persistence."""
    if bundle.clear_sky_index is None:
        return None
    geometry: dict[str, object] = {
        "clear_sky_index": bundle.clear_sky_index,
        "clear_future_pos": 0,
        "daylight_future_pos": 1 if bundle.daylight_index is not None else None,
        "target_mu": float(bundle.feature_mu[-1]),
        "target_sd": float(bundle.feature_sd[-1]),
        "clear_mu": float(bundle.feature_mu[bundle.clear_sky_index]),
        "clear_sd": float(bundle.feature_sd[bundle.clear_sky_index]),
        "clear_sky_training_max": float(
            bundle.raw[: bundle.train_end, bundle.clear_sky_index].max()
        ),
        "capacity": bundle.capacity,
    }
    if bundle.daylight_index is not None:
        geometry["daylight_mu"] = float(
            bundle.feature_mu[bundle.daylight_index]
        )
        geometry["daylight_sd"] = float(
            bundle.feature_sd[bundle.daylight_index]
        )
    return geometry


def _model_config(
    job: JobSpec,
    bundle: DatasetBundle,
    params: dict[str, object],
    gate: bool,
    ablation: frozenset[str] | set[str] | tuple[str, ...] | None = None,
    pc_fra: dict[str, object] | None = None,
):
    config = SimpleNamespace(
        seq_len=job.seq_len,
        pred_len=job.horizon,
        enc_in=bundle.values.shape[1],
        d_model=int(params.get("d_model", 64)),
        d_ff=max(128, 2 * int(params.get("d_model", 64))),
        n_heads=int(params.get("heads", 4)),
        e_layers=3,
        dropout=float(params.get("dropout", 0.1)),
        patch_hours=params.get("patch_hours", [1.0, 2.0, 4.0, 8.0]),
        gtr_period=int(params.get("gtr_period", 24)),
        corpatch_heads=int(params.get("heads", 4)),
        sample_hours=bundle.sample_hours,
        patchtst_d_model=int(params.get("patchtst_d_model", 128)),
        patchtst_d_ff=int(params.get("patchtst_d_ff", 256)),
        patchtst_heads=int(params.get("patchtst_heads", 16)),
        patchtst_layers=int(params.get("patchtst_layers", 3)),
        patchtst_dropout=float(params.get("patchtst_dropout", 0.2)),
        patchtst_patch_len=int(params.get("patchtst_patch_len", 16)),
        patchtst_stride=int(params.get("patchtst_stride", 8)),
        tcn_channels=int(params.get("tcn_channels", 128)),
        tcn_layers=int(params.get("tcn_layers", 3)),
        tcn_kernel_size=int(params.get("tcn_kernel_size", 3)),
        tcn_dropout=float(params.get("tcn_dropout", 0.1)),
        frets_embed_size=int(params.get("frets_embed_size", 128)),
        frets_hidden_size=int(params.get("frets_hidden_size", 256)),
        lstm_d_model=int(params.get("lstm_d_model", 128)),
        lstm_layers=int(params.get("lstm_layers", 1)),
        gru_d_model=int(params.get("gru_d_model", 128)),
        gru_layers=int(params.get("gru_layers", 1)),
        timexer_d_model=int(params.get("timexer_d_model", 256)),
        timexer_heads=int(params.get("timexer_heads", 4)),
        timexer_layers=int(params.get("timexer_layers", 3)),
        timexer_patch_len=int(params.get("timexer_patch_len", 6)),
        timexer_dropout=float(params.get("timexer_dropout", 0.1)),
        timexer_use_norm=bool(params.get("timexer_use_norm", False)),
    )
    config.pc_fra = dict(pc_fra) if pc_fra else None
    ablation_set = frozenset(ablation or ())
    if job.model == "psrc":
        psrc = psrc_settings(params, {"gate": gate})
        token_starts = _evenly_sample(bundle.train.starts, 6000)
        roles = resolve_physical_roles(bundle.feature_names)
        token_mean, token_scale = physical_token_statistics(
            bundle.values,
            token_starts,
            job.seq_len,
            roles,
            bundle.feature_mu,
            bundle.feature_sd,
            float(bundle.feature_mu[-1]),
            float(bundle.feature_sd[-1]),
        )
        # Ablation arm "no_gate" requires an unconditional correction, i.e.
        # an exact confidence gate of g == 1 (not the station's max_gate).
        use_gate = bool(psrc["gate"]) and "no_gate" not in ablation_set
        max_gate = 1.0 if "no_gate" in ablation_set else float(psrc["max_gate"])
        config.semantic_residual_max_correction = float(psrc["max_correction"])
        config.semantic_residual_max_gate = max_gate
        config.physical_semantic = {
            "roles": roles,
            "feature_mu": bundle.feature_mu.tolist(),
            "feature_sd": bundle.feature_sd.tolist(),
            "target_mu": float(bundle.feature_mu[-1]),
            "target_sd": float(bundle.feature_sd[-1]),
            "token_mean": token_mean.tolist(),
            "token_scale": token_scale.tolist(),
            "d_token": 32,
            "use_gate": use_gate,
        }
        config.ablation = tuple(sorted(ablation_set))
    if job.model == "smart_persistence":
        geometry = _solar_geometry(bundle)
        if geometry is None:
            raise ValueError(
                f"{job.dataset} has no deterministic clear-sky geometry "
                "feature for smart persistence"
            )
        config.solar_geometry = geometry
    return config


def _loader(
    bundle: DatasetBundle,
    starts: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    fm_embeddings: object | None = None,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        ForecastDataset(bundle, starts, fm_embeddings=fm_embeddings),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    bundle: DatasetBundle,
    device: torch.device,
    *,
    collect: bool = False,
) -> dict[str, object] | tuple[dict[str, object], dict[str, np.ndarray]]:
    """Evaluate a split.

    With ``collect=True`` the single evaluation pass also returns the
    per-window arrays in the model's standardized output space:
    ``pred_std`` (final forecast), ``base_std`` (numerical backbone
    forecast for PSRC, identical to ``pred_std`` otherwise) and
    ``gate`` (PSRC only). Arrays follow the loader's window order.
    """
    model.eval()
    predictions = []
    bases = []
    gates = []
    targets = []
    pc_psrcs = []
    pc_deltas = []
    with torch.no_grad():
        for batch in loader:
            target = batch[1]
            output = _forward_batch(model, batch, device)
            predictions.append(torch.as_tensor(output, device=device))
            targets.append(target)
            if collect:
                decomposition = getattr(model, "last_semantic_decomposition", None)
                if decomposition is not None:
                    bases.append(torch.as_tensor(decomposition[0], device=device))
                    gates.append(
                        torch.as_tensor(
                            getattr(model, "last_psrc_gate", None), device=device
                        )
                    )
                pc_residual = getattr(model, "last_pc_fra", None)
                if pc_residual is not None:
                    pc_psrc, pc_delta = pc_residual
                    pc_psrcs.append(torch.as_tensor(pc_psrc, device=device))
                    pc_deltas.append(torch.as_tensor(pc_delta, device=device))
    # Single host sync per evaluation pass: per-batch .cpu() transfers stall
    # badly when several training processes share one GPU.
    pred_std = torch.cat(predictions, dim=0).cpu().numpy()
    target_std = np.concatenate([t.numpy() for t in targets])
    target_sd = float(bundle.feature_sd[-1])
    target_mu = float(bundle.feature_mu[-1])
    pred_physical = np.maximum(0.0, pred_std * target_sd + target_mu)
    target_physical = target_std * target_sd + target_mu
    metrics = regression_metrics(pred_physical, target_physical, bundle.capacity)
    projected_std_error = (pred_physical - target_physical) / target_sd
    metrics["mse_standardized"] = float(np.mean(projected_std_error**2))
    metrics["mae_standardized"] = float(np.mean(np.abs(projected_std_error)))
    metrics["mbe_standardized"] = float(np.mean(projected_std_error))
    if not collect:
        return metrics
    if bases:
        base_std = torch.cat(bases, dim=0).cpu().numpy()
        gate_array = torch.cat(gates, dim=0).cpu().numpy()
    else:
        base_std = pred_std
        gate_array = None
    pc_psrc_array = torch.cat(pc_psrcs, dim=0).cpu().numpy() if pc_psrcs else None
    pc_delta_array = (
        torch.cat(pc_deltas, dim=0).cpu().numpy() if pc_deltas else None
    )
    arrays = {
        "pred_std": pred_std,
        "base_std": base_std,
        "gate": gate_array,
        "target_std": target_std,
        "pc_psrc": pc_psrc_array,
        "pc_delta": pc_delta_array,
    }
    return metrics, arrays


def _to_physical(arrays: dict[str, np.ndarray], bundle: DatasetBundle):
    """Apply the one authoritative projection used by ``regression_metrics``."""
    sd = float(bundle.feature_sd[-1])
    mu = float(bundle.feature_mu[-1])
    y = arrays["target_std"] * sd + mu
    yhat = np.maximum(0.0, arrays["pred_std"] * sd + mu)
    yhat0 = np.maximum(0.0, arrays["base_std"] * sd + mu)
    return y, yhat, yhat0


def _window_timestamps(loader: DataLoader, seq_len: int, bundle: DatasetBundle):
    starts = np.asarray(loader.dataset.starts, dtype=np.int64)
    target_starts = starts + seq_len
    return bundle.timestamps[target_starts].astype("datetime64[ns]").astype(np.int64)


def save_prediction_artifacts(
    directory: Path,
    arrays: dict[str, np.ndarray],
    loader: DataLoader,
    seq_len: int,
    bundle: DatasetBundle,
    metrics: dict[str, object],
) -> Path:
    """Persist per-window test arrays from the single headline evaluation."""
    y, yhat, yhat0 = _to_physical(arrays, bundle)
    delta = yhat - yhat0
    if arrays["gate"] is not None:
        gate = arrays["gate"].astype(np.float32)
    else:
        gate = np.full_like(yhat, np.nan, dtype=np.float32)
    window_start = _window_timestamps(loader, seq_len, bundle)
    if y.shape != gate.shape or window_start.shape != (y.shape[0],):
        raise ValueError("prediction artifact arrays have inconsistent shapes")
    # The dump must reproduce the authoritative headline metrics exactly.
    rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))
    mae = float(np.mean(np.abs(yhat - y)))
    if not np.isclose(rmse, float(metrics["rmse_physical"]), rtol=1e-6, atol=1e-8):
        raise AssertionError("dump RMSE disagrees with headline metrics")
    if not np.isclose(mae, float(metrics["mae_physical"]), rtol=1e-6, atol=1e-8):
        raise AssertionError("dump MAE disagrees with headline metrics")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "predictions.npz"
    np.savez_compressed(
        path,
        y=y.astype(np.float32),
        yhat0=yhat0.astype(np.float32),
        yhat_psrc=yhat.astype(np.float32),
        delta=delta.astype(np.float32),
        gate=gate,
        window_start=window_start.astype(np.int64),
    )
    return path


def save_checkpoint(directory: Path, model: torch.nn.Module) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "checkpoint.pt"
    torch.save({"state_dict": model.state_dict()}, path)
    return path


def _parameter_record(
    job: JobSpec, config: ExperimentConfig
) -> tuple[str, dict[str, object]]:
    if job.model in PUBLIC_BASELINE_MODELS:
        # Locked training protocol + architecture hyperparameters published
        # by the reference repository (no per-site/per-seed retuning).
        return "public_configuration", {
            **dict(config.baseline),
            **dict(config.public_baselines[job.model]),
        }
    if job.model != "psrc":
        return "baseline_fixed", dict(config.baseline)
    record = load_best_parameters(config.best_parameters)[job.dataset]
    return str(record["source"]), dict(record["params"])


def _train_model(
    job: JobSpec,
    config: ExperimentConfig,
    bundle: DatasetBundle,
    params: dict[str, object],
    psrc: dict[str, object],
    *,
    epochs: int | None = None,
    patience: int | None = None,
    epoch_callback=None,
    ablation: frozenset[str] | set[str] | tuple[str, ...] | None = None,
    fm_embeddings: object | None = None,
    initial_state_dict: dict[str, torch.Tensor] | None = None,
    train_pc_only: bool = False,
    pc_fra_config: dict[str, object] | None = None,
):
    """Train a trainable model using the train/validation splits only.

    Returns ``(model, history, stopper, batch_size)`` with the best
    validation checkpoint restored. ``epoch_callback(epoch, val_score)``
    runs after each validation pass and may raise to abort the trial
    (used by the Optuna tuning study for pruning); the test split is never
    touched here.
    """
    epochs = int(config.epochs if epochs is None else epochs)
    patience = int(config.patience if patience is None else patience)
    ablation_set = frozenset(ablation or ())
    _seed_everything(job.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fm_kind = getattr(fm_embeddings, "kind", None) if fm_embeddings else None
    if fm_kind == "pc_fra":
        if pc_fra_config is None:
            raise ValueError("pc_fra pack requires pc_fra_config")
        fm_pc_cfg = dict(pc_fra_config)
    else:
        fm_pc_cfg = None
    model = build_model(
        job.model,
        _model_config(
            job,
            bundle,
            params,
            bool(config.psrc["gate"]),
            ablation=ablation_set,
            pc_fra=fm_pc_cfg,
        ),
    ).to(device)
    if initial_state_dict is not None:
        incompatible = model.load_state_dict(initial_state_dict, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing = [
            name
            for name in incompatible.missing_keys
            if not name.startswith("pc_fra_adapter.")
        ]
        if unexpected or missing:
            raise ValueError(
                f"incompatible warm start: missing={missing}, unexpected={unexpected}"
            )
    if train_pc_only:
        if model.pc_fra_adapter is None:
            raise ValueError("train_pc_only requires the pc_fra residual adapter")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("pc_fra_adapter."))
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError(f"{job.model} unexpectedly has no trainable parameters")
    optimizer_class = (
        torch.optim.AdamW
        if str(params.get("optimizer", "adam")) == "adamw"
        else torch.optim.Adam
    )
    optimizer = optimizer_class(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(params.get("lr", 1e-3)),
        weight_decay=float(params.get("weight_decay", 1e-4)),
    )
    batch_size = int(params.get("batch", 128))
    scheduler = _make_scheduler(params, optimizer, epochs)
    validation_loader = _loader(
        bundle,
        bundle.validation.starts,
        batch_size,
        False,
        job.seed,
        fm_embeddings=fm_embeddings,
    )
    stopper = EarlyStopping(patience)
    best_state = None
    history = []
    for epoch in range(1, epochs + 1):
        if train_pc_only:
            # Frozen PSRC in eval mode; only the PC-FRA adapter learns.
            model.eval()
            model.pc_fra_adapter.train()
        else:
            model.train()
        starts = _training_starts(
            bundle, params, config.train_windows_per_epoch, job.seed, epoch
        )
        train_loader = _loader(
            bundle,
            starts,
            batch_size,
            True,
            job.seed + epoch,
            fm_embeddings=fm_embeddings,
        )
        losses = []
        for batch in train_loader:
            target = batch[1].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = _forward_batch(model, batch, device)
            if train_pc_only:
                # PC-FRA stage 1: adapter learns the residual remaining
                # after frozen PSRC, Huber(epsilon*tanh(raw), y - y_psrc).
                # Optional correction-tuning regularisers are off unless
                # pc_fra_config requests them (defaults reproduce the
                # registered behaviour bitwise):
                #   huber_beta    Huber transition (std units), default 1.0
                #   energy_weight lambda * mean(Delta^2)
                #   smooth_weight lambda * mean((Delta_h - Delta_{h-1})^2)
                pc_pair = model.last_pc_fra
                if pc_pair is None:
                    raise RuntimeError("pc_fra adapter produced no residual output")
                psrc_std, delta_std = pc_pair
                residual_target = target - psrc_std
                huber_beta = float(fm_pc_cfg.get("huber_beta", 1.0))
                loss = F.smooth_l1_loss(
                    delta_std, residual_target, beta=huber_beta,
                    reduction="mean",
                )
                energy_weight = float(fm_pc_cfg.get("energy_weight", 0.0))
                if energy_weight > 0.0:
                    loss = loss + energy_weight * (delta_std ** 2).mean()
                smooth_weight = float(fm_pc_cfg.get("smooth_weight", 0.0))
                if smooth_weight > 0.0 and delta_std.shape[1] > 1:
                    loss = loss + smooth_weight * (
                        (delta_std[:, 1:] - delta_std[:, :-1]) ** 2
                    ).mean()
            else:
                loss = point_loss(
                    prediction,
                    target,
                    bundle,
                    str(params.get("loss_base", "mse")),
                    float(params.get("huber_delta", 1.0)),
                    float(params.get("daylight_weight", 0.0)),
                    float(params.get("ramp_weight", 0.0)),
                )
            # The "numerical_backbone" ablation emits the base forecast only;
            # it is trained with the point loss alone (no residual/anchor
            # auxiliary terms, since no correction is produced or used).
            if (
                job.model == "psrc"
                and "numerical_backbone" not in ablation_set
                and not train_pc_only
            ):
                base, correction = model.last_semantic_decomposition
                loss = loss + float(psrc["semantic_loss_weight"]) * (
                    psrc_residual_loss(
                        base,
                        correction,
                        target,
                        alpha=float(psrc["residual_alpha"]),
                    )
                )
                loss = loss + float(psrc["base_anchor_weight"]) * point_loss(
                    base,
                    target,
                    bundle,
                    str(params.get("loss_base", "mse")),
                    float(params.get("huber_delta", 1.0)),
                    float(params.get("daylight_weight", 0.0)),
                    float(params.get("ramp_weight", 0.0)),
                )
            loss.backward()
            optimizer.step()
            # Keep the scalar on-device: a per-batch host sync serialises all
            # contending processes on a shared GPU. Sync once after the epoch.
            losses.append(loss.detach())
        if scheduler is not None:
            scheduler.step()
        validation = _evaluate(model, validation_loader, bundle, device)
        val_score = float(validation["mse_standardized"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(torch.stack(losses).mean().item()),
                "validation_mse_standardized": val_score,
            }
        )
        if val_score < stopper.best_score:
            best_state = deepcopy(model.state_dict())
        if epoch_callback is not None:
            epoch_callback(epoch, val_score)
        if stopper.update(val_score):
            break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    return model, history, stopper, batch_size


def run_job(
    job: JobSpec,
    config: ExperimentConfig,
    artifacts_dir: Path | None = None,
) -> dict[str, object]:
    """Run one formal task. The caller owns serialization and resume logic.

    When ``artifacts_dir`` is provided, the single headline test evaluation
    also persists ``predictions.npz`` (every model) and ``checkpoint.pt``
    (trainable models) into that directory.
    """
    started = time.perf_counter()
    parameter_source, params = _parameter_record(job, config)
    psrc = psrc_settings(params, config.psrc)
    bundle = load_dataset(
        config.dataset(job.dataset).path,
        job.seq_len,
        job.horizon,
        config.train_fraction,
        config.validation_end_fraction,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if job.model in DETERMINISTIC_MODELS:
        _seed_everything(job.seed)
        model = build_model(
            job.model,
            _model_config(job, bundle, params, bool(config.psrc["gate"])),
        ).to(device)
        batch_size = int(params.get("batch", 128))
        test_loader = _loader(bundle, bundle.test.starts, batch_size, False, job.seed)
        test_metrics, test_arrays = _evaluate(
            model, test_loader, bundle, device, collect=True
        )
        prediction_artifact = None
        if artifacts_dir is not None:
            prediction_artifact = save_prediction_artifacts(
                artifacts_dir,
                test_arrays,
                test_loader,
                job.seq_len,
                bundle,
                test_metrics,
            ).name
        return {
            "status": "complete",
            "job": {
                "key": job.key,
                "dataset": job.dataset,
                "model": job.model,
                "seq_len": job.seq_len,
                "horizon": job.horizon,
                "seed": job.seed,
            },
            "dataset_path": str(bundle.path),
            "dataset_sha256": hashlib.sha256(
                bundle.path.read_bytes()
            ).hexdigest(),
            "parameter_source": "deterministic_baseline",
            "parameters": {},
            "epochs_requested": config.epochs,
            "epochs_ran": 0,
            "best_epoch": 0,
            "early_stopped": False,
            "stop_reason": "deterministic_baseline",
            "patience": config.patience,
            "train_end": bundle.train_end,
            "validation_end": bundle.validation_end,
            "capacity": bundle.capacity,
            "capacity_source": "training_empirical_max",
            "sample_hours": bundle.sample_hours,
            "features": list(bundle.feature_names),
            "test": test_metrics,
            "psrc_gate_mean": None,
            "headline_test_evaluation_passes": 1,
            "predictions_artifact": prediction_artifact,
            "checkpoint_artifact": None,
            "training_history": [],
            "seconds": time.perf_counter() - started,
        }

    model, history, stopper, batch_size = _train_model(
        job, config, bundle, params, psrc
    )
    test_loader = _loader(bundle, bundle.test.starts, batch_size, False, job.seed)
    test_metrics, test_arrays = _evaluate(
        model, test_loader, bundle, device, collect=True
    )
    gate_array = test_arrays["gate"]
    prediction_artifact = None
    checkpoint_artifact = None
    if artifacts_dir is not None:
        prediction_artifact = save_prediction_artifacts(
            artifacts_dir,
            test_arrays,
            test_loader,
            job.seq_len,
            bundle,
            test_metrics,
        ).name
        checkpoint_artifact = save_checkpoint(artifacts_dir, model).name
    result = {
        "status": "complete",
        "job": {
            "key": job.key,
            "dataset": job.dataset,
            "model": job.model,
            "seq_len": job.seq_len,
            "horizon": job.horizon,
            "seed": job.seed,
        },
        "dataset_path": str(bundle.path),
        "dataset_sha256": hashlib.sha256(bundle.path.read_bytes()).hexdigest(),
        "parameter_source": parameter_source,
        "parameters": params,
        "epochs_requested": config.epochs,
        "epochs_ran": len(history),
        "best_epoch": stopper.best_epoch,
        "early_stopped": len(history) < config.epochs,
        "stop_reason": "patience" if len(history) < config.epochs else "epoch_budget",
        "patience": config.patience,
        "train_end": bundle.train_end,
        "validation_end": bundle.validation_end,
        "capacity": bundle.capacity,
        "capacity_source": "training_empirical_max",
        "sample_hours": bundle.sample_hours,
        "features": list(bundle.feature_names),
        "test": test_metrics,
        "psrc_gate_mean": (
            float(np.mean(gate_array)) if gate_array is not None else None
        ),
        "headline_test_evaluation_passes": 1,
        "predictions_artifact": prediction_artifact,
        "checkpoint_artifact": checkpoint_artifact,
        "training_history": history,
        "seconds": time.perf_counter() - started,
    }
    return result
