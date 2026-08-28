"""Common-protocol PV benchmark against recent-paper models.

The runner deliberately uses one chronological split, one feature tensor and
one training budget for every trainable model.  It imports AMPDNet from the
authors' unmodified 2026 Solar Energy repository under ``external/AMPDNet``;
the repository's dormant TSLANet dependency is stubbed because the published
ConvPVNet/AMPDNet forward path does not instantiate that spectral block.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

# 允许从当前文件夹直接运行本脚本，同时固定所有本地模块的解析根目录。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from rp_gtr_ablation import RPGTR as AblationRPGTR, load_windows, seed_all
from models.ours_v73 import Model as LegacyOursModel
from models.ours_v73 import NWPConditionedForecastDecoder
from models.factory import build_model as build_local_model
from utils.prompt_features import (build_or_load_prompt_features,
                                   fit_prompt_regime_thresholds)
from utils.channel_semantics import (controlled_channel_semantics,
                                     load_channel_role_signatures,
                                     load_channel_semantics)


AMPD_ROOT = ROOT / "external" / "AMPDNet"
CROSS_UNET_ROOT = ROOT / "external" / "PV-power"


def _load_cross_unet_module(name: str):
    """Load one released Cross-Unet module without the top-level ``models`` name.

    AMPDNet and Cross-Unet both publish a package called ``models``.  The
    common-protocol runner evaluates several baselines in one Python process,
    so ordinary imports make the second repository silently reuse the first
    repository's package.  Loading the requested file under a private module
    name preserves the authors' code while keeping the repositories isolated.
    """
    cross_root = str(CROSS_UNET_ROOT)
    module_path = CROSS_UNET_ROOT / "models" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(
        f"_released_cross_unet_{name}", module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load released module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    # Both author repositories also publish a top-level package named
    # ``layers``.  Temporarily isolate that namespace while executing the
    # Cross-Unet file, then restore the already-loaded AMPDNet namespace.  The
    # released modules import the required classes/functions by value, so the
    # restored package cannot alter their forward pass.
    saved_layers = {
        key: value for key, value in list(sys.modules.items())
        if key == "layers" or key.startswith("layers.")
    }
    for key in saved_layers:
        sys.modules.pop(key, None)
    # Cross-Unet ships ``layers`` as a namespace directory (no __init__.py),
    # whereas AMPDNet ships a regular package with the same name.  Python
    # otherwise prefers the later regular package even when Cross-Unet is
    # first on sys.path.  Pin the namespace path explicitly for this import.
    cross_layers = types.ModuleType("layers")
    cross_layers.__path__ = [str(CROSS_UNET_ROOT / "layers")]
    cross_layers.__package__ = "layers"
    sys.modules["layers"] = cross_layers
    old_path = list(sys.path)
    sys.path[:] = [cross_root] + [p for p in sys.path if p != cross_root]
    try:
        spec.loader.exec_module(module)
    finally:
        for key in list(sys.modules):
            if key == "layers" or key.startswith("layers."):
                sys.modules.pop(key, None)
        sys.modules.update(saved_layers)
        sys.path[:] = old_path
    return module


class WindowSet(Dataset):
    def __init__(self, x, starts, seq_len, horizon, cycle_len, cycle_ids,
                 target_observed=None, include_future=False, future_indices=None,
                 prompt_by_start=None, prompt_shuffle=False):
        self.x, self.starts = x, starts
        self.seq_len, self.horizon, self.cycle_len = seq_len, horizon, cycle_len
        self.cycle_ids = cycle_ids
        self.include_future = include_future
        self.future_indices = tuple(int(i) for i in (future_indices or []))
        self.target_observed = (np.ones(len(x), dtype=np.float32) if target_observed is None
                                else np.asarray(target_observed, dtype=np.float32))
        self.prompt_by_start = prompt_by_start
        self.prompt_shuffle = bool(prompt_shuffle)
        if self.prompt_shuffle:
            self.prompt_starts = np.random.default_rng(2026).permutation(
                np.asarray(self.starts, dtype=np.int64)
            )
        else:
            self.prompt_starts = np.asarray(self.starts, dtype=np.int64)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = int(self.starts[i])
        cycle = int(self.cycle_ids[s]) if self.cycle_ids is not None else int(s % self.cycle_len)
        x = torch.tensor(self.x[s:s + self.seq_len])
        y = torch.tensor(self.x[s + self.seq_len:s + self.seq_len + self.horizon, -1])
        y_mask = torch.tensor(self.target_observed[s + self.seq_len:s + self.seq_len + self.horizon])
        prompt = None
        if self.prompt_by_start is not None:
            prompt_s = int(self.prompt_starts[i])
            prompt = torch.tensor(self.prompt_by_start[prompt_s])
        if self.include_future:
            future_x = self.x[s + self.seq_len:s + self.seq_len + self.horizon]
            if self.future_indices:
                # Audited NWP decoder path: the target and quality channels do
                # not exist in this tensor, rather than merely being masked.
                future_x = future_x[:, self.future_indices].copy()
            else:
                # Legacy M21 path retained only for reproducible ablation.
                future_x = future_x.copy()
                future_x[:, -1] = 0.0
            result = (x, y, y_mask, torch.tensor(cycle), torch.tensor(future_x))
            return result + ((prompt,) if prompt is not None else ())
        if prompt is not None:
            # Keep prompt-only batches structurally distinct from the legacy
            # five-field future-covariate batch. The empty tensor is a typed
            # placeholder and contains no forecast-interval information.
            empty_future = torch.empty((self.horizon, 0), dtype=x.dtype)
            return x, y, y_mask, torch.tensor(cycle), empty_future, prompt
        return x, y, y_mask, torch.tensor(cycle)


class EpochStratifiedSampler(Sampler[int]):
    """Use a new chronology-covering training subset on every epoch.

    One index is sampled from every equal-width chronological bin.  The sample
    budget per epoch is therefore unchanged, while repeated epochs can expose
    the backbone to more of a long training record than a single fixed subset.
    """

    def __init__(self, pool_size: int, sample_count: int, seed: int):
        self.pool_size = int(pool_size)
        self.sample_count = min(int(sample_count), self.pool_size)
        self.seed = int(seed)
        self.epoch = 0
        if self.pool_size < 1 or self.sample_count < 1:
            raise ValueError("epoch-stratified sampling requires a non-empty pool")

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        edges = np.linspace(
            0, self.pool_size, self.sample_count + 1, dtype=np.int64,
        )
        selected = np.empty(self.sample_count, dtype=np.int64)
        for index in range(self.sample_count):
            low = int(edges[index])
            high = max(low + 1, int(edges[index + 1]))
            selected[index] = int(rng.integers(low, high))
        rng.shuffle(selected)
        return iter(selected.tolist())

    def __len__(self):
        return self.sample_count


class MaskedHistoryReconstructionHead(nn.Module):
    """Temporary training-only decoder for masked historical reconstruction.

    The forecasting encoder returns one token grid per input variable.  This
    small shared projection maps every token back to one scalar and interpolates
    the token axis to the original look-back length.  The module is discarded
    before forecasting fine-tuning and is never part of inference.
    """

    def __init__(self, channels: int, d_model: int, seq_len: int):
        super().__init__()
        self.channels = int(channels)
        self.seq_len = int(seq_len)
        self.projection = nn.Linear(int(d_model), 1)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        if encoded.ndim != 3:
            raise ValueError(
                f"masked reconstruction expects [B*C,N,D], got {tuple(encoded.shape)}"
            )
        if encoded.shape[0] % self.channels:
            raise ValueError(
                "encoded batch is not divisible by the configured channel count"
            )
        batch = encoded.shape[0] // self.channels
        token_values = self.projection(encoded).squeeze(-1)
        token_values = token_values.reshape(batch, self.channels, -1)
        reconstructed = F.interpolate(
            token_values, size=self.seq_len, mode="linear", align_corners=False,
        )
        return reconstructed.transpose(1, 2)


def build_multiscale_history_mask(
        x: torch.Tensor, channel_indices, span_lengths) -> torch.Tensor:
    """Mask one contiguous physical-scale span per sample.

    The function uses only the historical tensor shape and the already-defined
    patch scales.  It never inspects the future target.  Standardised zero is
    the training-mean replacement value, so no validation or test statistic is
    introduced.
    """
    if x.ndim != 3:
        raise ValueError(f"expected history [B,L,C], got {tuple(x.shape)}")
    indices = torch.as_tensor(
        channel_indices, device=x.device, dtype=torch.long,
    )
    indices = indices[(indices >= 0) & (indices < x.shape[-1])]
    if indices.numel() == 0:
        raise ValueError("masked pretraining requires at least one valid channel")
    spans = torch.as_tensor(
        sorted({max(1, min(int(span), x.shape[1])) for span in span_lengths}),
        device=x.device, dtype=torch.long,
    )
    if spans.numel() == 0:
        raise ValueError("masked pretraining requires at least one span length")
    mask = torch.zeros_like(x, dtype=torch.bool)
    selected = spans[torch.randint(spans.numel(), (x.shape[0],), device=x.device)]
    for sample, span in enumerate(selected.tolist()):
        start = int(torch.randint(
            max(1, x.shape[1] - span + 1), (1,), device=x.device,
        ).item())
        mask[sample, start:start + span, indices] = True
    return mask


def run_masked_history_pretraining(
        model: nn.Module, loader: DataLoader, device: torch.device, args) -> dict:
    """Pretrain the unchanged numerical backbone on training histories only."""
    epochs = int(args.masked_pretrain_epochs)
    if epochs <= 0:
        return {
            "enabled": False,
            "epochs": 0,
            "epoch_losses": [],
            "span_lengths": [],
            "channel_indices": [],
        }
    encoder = getattr(model, "encoder", None)
    patch = getattr(model, "patch", None)
    if encoder is None or patch is None:
        raise ValueError("masked pretraining requires the existing patch encoder")
    spans = [int(patch_len) for patch_len, _ in getattr(patch, "scales", [])]
    if not spans:
        spans = [max(2, int(args.patch_len))]
    frequency_indices = getattr(model, "frequency_indices", None)
    if frequency_indices is None:
        channel_indices = list(range(int(getattr(model, "channels", 0))))
    else:
        channel_indices = [
            int(index) for index in frequency_indices.detach().cpu().tolist()
        ]
    channels = int(getattr(model, "channels", 0))
    if channels <= 0:
        raise ValueError("masked pretraining could not infer the input channels")
    head = MaskedHistoryReconstructionHead(
        channels, int(args.d_model), int(args.seq_len),
    ).to(device)
    capture = {}

    def capture_encoder_output(_module, _inputs, output):
        capture["encoded"] = output

    handle = encoder.register_forward_hook(capture_encoder_output)
    previous_semantic = bool(getattr(model, "semantic_enabled", False))
    previous_fsra = bool(getattr(model, "fsra_enabled", False))
    if hasattr(model, "semantic_enabled"):
        model.semantic_enabled = False
    if hasattr(model, "fsra_enabled"):
        model.fsra_enabled = False
    backbone_params = [parameter for parameter in model.parameters()
                       if parameter.requires_grad]
    optimizer = torch.optim.Adam(
        backbone_params + list(head.parameters()),
        lr=float(args.masked_pretrain_lr),
        weight_decay=float(args.weight_decay),
    )
    epoch_losses = []
    try:
        for _epoch in range(epochs):
            model.train()
            head.train()
            total_loss = 0.0
            total_batches = 0
            for batch in loader:
                xb, _yb, _ymb, cb, fxb, prompt_x = unpack_batch(batch, device)
                history_mask = build_multiscale_history_mask(
                    xb, channel_indices, spans,
                )
                corrupted = xb.masked_fill(history_mask, 0.0)
                capture.clear()
                optimizer.zero_grad(set_to_none=True)
                predict(model, corrupted, cb, fxb, prompt_x)
                if "encoded" not in capture:
                    raise RuntimeError("forecast encoder hook did not receive tokens")
                reconstructed = head(capture["encoded"])
                value_error = (reconstructed - xb).square()
                value_loss = value_error[history_mask].mean()
                delta_mask = history_mask[:, 1:] & history_mask[:, :-1]
                if delta_mask.any():
                    reconstructed_delta = reconstructed[:, 1:] - reconstructed[:, :-1]
                    target_delta = xb[:, 1:] - xb[:, :-1]
                    delta_loss = (
                        reconstructed_delta - target_delta
                    ).square()[delta_mask].mean()
                else:
                    delta_loss = value_loss.new_zeros(())
                loss = (
                    value_loss
                    + float(args.masked_pretrain_delta_weight) * delta_loss
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    backbone_params + list(head.parameters()), 1.0,
                )
                optimizer.step()
                total_loss += float(loss.detach())
                total_batches += 1
            epoch_losses.append(total_loss / max(1, total_batches))
    finally:
        handle.remove()
        if hasattr(model, "semantic_enabled"):
            model.semantic_enabled = previous_semantic
        if hasattr(model, "fsra_enabled"):
            model.fsra_enabled = previous_fsra
    return {
        "enabled": True,
        "epochs": epochs,
        "epoch_losses": epoch_losses,
        "span_lengths": sorted(set(spans)),
        "channel_indices": channel_indices,
        "delta_weight": float(args.masked_pretrain_delta_weight),
        "learning_rate": float(args.masked_pretrain_lr),
        "temporary_head_discarded": True,
    }


class TargetFirst(nn.Module):
    """Move our canonical last target channel to index 0 for author models."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, cycle=None):
        x = torch.cat([x[..., -1:], x[..., :-1]], dim=-1)
        return self.model(x)


class CrossUnetOfficialAdapter(nn.Module):
    """Common-protocol adapter around the authors' released Cross-Unet.

    The official architecture requires equal historical and forward-weather
    lengths.  We therefore use the last H historical steps and the H valid NWP
    steps, exactly matching the paper's one-day encoder-decoder construction.
    An earlier H-step historical block is used only for the leakage-free P-Corr
    approximation.  No future PV or target-availability flag is supplied.
    """
    def __init__(self, horizon: int, stats):
        super().__init__()
        history_idx = stats.get("core_history_indices") or []
        nwp_hist_idx = stats.get("core_nwp_history_indices") or []
        future_names = stats.get("future_features") or []
        if len(history_idx) != 7 or len(nwp_hist_idx) != 6 or len(future_names) != 6:
            raise ValueError(
                "Cross-Unet requires the six-channel --future-feature-policy gefcom_nwp_core"
            )
        self.horizon = horizon
        self.register_buffer("history_indices", torch.tensor(history_idx, dtype=torch.long), persistent=False)
        self.register_buffer("nwp_history_indices", torch.tensor(nwp_hist_idx, dtype=torch.long), persistent=False)
        CrossUnetModel = _load_cross_unet_module("Cross_Unet").Model

        seg_len = 12 if horizon < 49 else (24 if horizon <= 384 else 48)
        config = SimpleNamespace(
            enc_in=13, seq_len=horizon, pred_len=horizon, seg_len=seg_len,
            task_name="long_term_forecast", e_layers=3, n_heads=4,
            d_model=256, d_ff=512, factor=10, dropout=0.05,
            useweather=True, usenonlinearproject=True, usebottle=True,
            convmerge=False, swichchannel=False, twofilter=True,
        )
        self.model = CrossUnetModel(config)
        self.recommended_lr = 1e-4
        self.history_features_used = [stats["feature_names"][i] for i in history_idx]
        self.future_features_used = list(future_names)
        self.protocol_note = "official Cross-Unet, H-step history + H-step NWP, prior H-step P-Corr"

    def forward(self, x, cycle=None, future_x=None):
        if future_x is None:
            raise ValueError("Cross-Unet requires audited future NWP")
        h = self.horizon
        if x.shape[1] < 2 * h:
            raise ValueError(f"Cross-Unet P-Corr requires look-back >= {2*h}, got {x.shape[1]}")
        current = x[:, -h:].index_select(-1, self.history_indices)
        previous = x[:, -2*h:-h].index_select(-1, self.history_indices)
        historical_nwp = x[:, -h:].index_select(-1, self.nwp_history_indices)
        out = self.model(current, None, future_x, None, historical_nwp, previous)
        return out[:, :h, -1]


class NWPOnlyTransformer(nn.Module):
    """Strong future-forcing control without any historical memory."""
    def __init__(self, future_channels: int, horizon: int, d_model: int):
        super().__init__()
        self.horizon = horizon
        self.embed = nn.Linear(future_channels, d_model)
        self.lead = nn.Embedding(horizon, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, 4 if d_model % 4 == 0 else 1, 4 * d_model,
            dropout=0.1, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, cycle=None, future_x=None):
        if future_x is None:
            raise ValueError("NWP-only Transformer requires audited future NWP")
        lead = torch.arange(self.horizon, device=future_x.device)
        z = self.embed(future_x) + self.lead(lead).unsqueeze(0)
        return self.head(self.encoder(z)).squeeze(-1)


class VanillaHistoryNWPDecoder(nn.Module):
    """Vanilla history memory with the same NWP decoder used by M22."""
    def __init__(self, history_indices, future_channels: int, seq_len: int,
                 horizon: int, d_model: int, feature_names):
        super().__init__()
        self.register_buffer("history_indices", torch.tensor(history_indices, dtype=torch.long), persistent=False)
        self.history_embed = nn.Linear(len(history_indices), d_model)
        self.position = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, 4 if d_model % 4 == 0 else 1, 4 * d_model,
            dropout=0.1, activation="gelu", batch_first=True,
        )
        self.history_encoder = nn.TransformerEncoder(layer, 2)
        self.decoder = NWPConditionedForecastDecoder(future_channels, horizon, d_model)
        self.history_features_used = [feature_names[i] for i in history_indices]

    def forward(self, x, cycle=None, future_x=None):
        if future_x is None:
            raise ValueError("vanilla NWP decoder requires audited future NWP")
        hist = x.index_select(-1, self.history_indices)
        memory = self.history_encoder(self.history_embed(hist) + self.position[:, :hist.shape[1]])
        return self.decoder(memory, future_x, memory.mean(dim=1))


class HistorySelectWrapper(nn.Module):
    """Select a common historical feature set before the proposed encoder."""
    def __init__(self, model: nn.Module, indices, feature_names):
        super().__init__()
        self.model = model
        self.register_buffer("indices", torch.tensor(indices, dtype=torch.long), persistent=False)
        self.history_features_used = [feature_names[i] for i in indices]

    def forward(self, x, cycle=None, future_x=None):
        return self.model(x.index_select(-1, self.indices), cycle, future_x)

    @property
    def last_scale_weights(self): return self.model.last_scale_weights
    @property
    def last_tf_weights(self): return self.model.last_tf_weights
    @property
    def last_output_correction(self): return self.model.last_output_correction
    @property
    def route_scale_weights(self): return self.model.route_scale_weights
    @property
    def route_tf_weights(self): return self.model.route_tf_weights


class ReleasedHistoryBaseline(nn.Module):
    """Adapter for historical-only baselines released with Cross-Unet."""
    def __init__(self, kind: str, args, channels: int):
        super().__init__()
        if str(CROSS_UNET_ROOT) not in sys.path:
            sys.path.insert(0, str(CROSS_UNET_ROOT))
        common = dict(
            seq_len=args.seq_len, pred_len=args.horizon, enc_in=channels,
            c_out=channels, useweather=False,
        )
        self.kind = kind
        if kind == "cyclenet":
            Model = _load_cross_unet_module("CycleNet").Model
            config = SimpleNamespace(
                **common, cycle=args.cycle_len, model_type="mlp",
                d_model=256, use_revin=1,
            )
            self.model = Model(config)
            self.recommended_lr = 1e-4
        elif kind == "patchmlp":
            Model = _load_cross_unet_module("PatchMLP").Model
            patch_array = ([48, 24, 12, 6] if args.seq_len >= 96
                           else [16, 8, 4, 2])
            config = SimpleNamespace(
                **common, output_attention=False, use_norm=True,
                d_model=max(64, args.d_model), patch_len_arryay=patch_array,
                e_layers=3,
            )
            self.model = Model(config)
            self.recommended_lr = 1e-4
        elif kind == "timemixer":
            Model = _load_cross_unet_module("TimeMixer").Model
            config = SimpleNamespace(
                **common, task_name="long_term_forecast", label_len=args.horizon,
                down_sampling_window=2, down_sampling_layers=3,
                down_sampling_method="avg", channel_independence=0,
                e_layers=3, moving_avg=25, use_future_temporal_feature=0,
                d_model=max(64, args.d_model), d_ff=max(256, 4 * args.d_model),
                dropout=0.1, embed="fixed", freq="h", use_norm=1, top_k=5,
                decomp_method="moving_avg",
            )
            self.model = Model(config)
            self.recommended_lr = 1e-3
        elif kind == "timesnet":
            Model = _load_cross_unet_module("TimesNet").Model
            config = SimpleNamespace(
                **common, task_name="long_term_forecast",
                label_len=args.horizon, top_k=3, num_kernels=4,
                e_layers=2, d_model=max(32, args.d_model),
                d_ff=max(64, 2 * args.d_model), embed="timeF",
                freq="h", dropout=0.1,
            )
            self.model = Model(config)
            self.recommended_lr = 1e-3
        else:
            raise ValueError(f"unsupported released baseline: {kind}")

    def forward(self, x, cycle=None):
        if self.kind == "cyclenet":
            out = self.model(x, cycle, None)
        elif self.kind == "patchmlp":
            out = self.model(x, None, None)
        elif self.kind == "timesnet":
            # The released TimesNet wrapper requires an explicit four-column
            # hourly time-marker tensor even when timestamps are unavailable.
            # Calendar/solar clocks are already present in ``x``; zeros here
            # avoid inventing a second, inconsistent calendar signal.
            time_mark = x.new_zeros(x.shape[0], x.shape[1], 4)
            out = self.model(x, time_mark, None)
        else:
            out = self.model(x, None, None, None)
        return out[..., -1]


class Persistence(nn.Module):
    def forward(self, x, cycle=None):
        return x[:, -1:, -1]


class SolarTrajectoryPersistence(nn.Module):
    """Smart persistence using deterministic future clear-sky geometry only."""
    def __init__(self, stats):
        super().__init__()
        clear_name = stats.get("clear_sky_feature")
        future_names = stats.get("solar_future_features") or []
        if clear_name is None or clear_name not in future_names:
            raise ValueError("solar trajectory persistence requires a clear-sky geometry feature")
        self.clear_hist_idx = int(stats["clear_sky_index"])
        self.clear_future_pos = future_names.index(clear_name)
        daylight_name = stats.get("daylight_feature")
        self.daylight_future_pos = (future_names.index(daylight_name)
                                    if daylight_name in future_names else None)
        self.target_mu = float(stats["target_mu"])
        self.target_sd = float(stats["target_sd"])
        self.clear_mu = float(stats["feature_mu"][self.clear_hist_idx])
        self.clear_sd = float(stats["feature_sd"][self.clear_hist_idx])
        self.clear_max = max(float(stats["clear_sky_training_max"]), 1e-6)
        self.capacity = max(float(stats["capacity"]), 1e-6)
        if self.daylight_future_pos is not None:
            daylight_hist_idx = stats["feature_names"].index(daylight_name)
            self.daylight_mu = float(stats["feature_mu"][daylight_hist_idx])
            self.daylight_sd = float(stats["feature_sd"][daylight_hist_idx])

    def forward(self, x, cycle=None, future_x=None):
        if future_x is None:
            raise ValueError("solar trajectory persistence requires known future solar geometry")
        power = x[..., -1] * self.target_sd + self.target_mu
        clear_hist = (x[..., self.clear_hist_idx] * self.clear_sd + self.clear_mu).clamp_min(0.0)
        clear_future = (future_x[..., self.clear_future_pos] * self.clear_sd + self.clear_mu).clamp_min(0.0)
        clear_norm = clear_hist / self.clear_max
        valid = clear_norm > 0.02
        positions = torch.arange(x.shape[1], device=x.device).view(1, -1)
        latest = torch.where(valid, positions, positions.new_full(positions.shape, -1)).max(dim=1).values
        safe_latest = latest.clamp_min(0)
        kpv_series = (power / self.capacity) / clear_norm.clamp_min(0.02)
        kpv = kpv_series.gather(1, safe_latest[:, None]).squeeze(1)
        kpv = torch.where(latest >= 0, kpv, torch.zeros_like(kpv)).clamp(0.0, 1.5)
        base_phys = self.capacity * kpv[:, None] * (clear_future / self.clear_max)
        if self.daylight_future_pos is not None:
            daylight = (future_x[..., self.daylight_future_pos] * self.daylight_sd
                        + self.daylight_mu)
            base_phys = base_phys * (daylight > 0.5).to(base_phys.dtype)
        return (base_phys - self.target_mu) / self.target_sd


class SeasonalNaive(nn.Module):
    def __init__(self, horizon, cycle_len):
        super().__init__()
        self.horizon, self.cycle_len = horizon, cycle_len

    def forward(self, x, cycle=None):
        if x.shape[1] >= self.cycle_len:
            start = x.shape[1] - self.cycle_len
            # Repeat the complete most recent daily shape.  Clamping here
            # would silently turn every lead beyond one cycle into persistence.
            idx = start + torch.arange(self.horizon, device=x.device) % self.cycle_len
            return x.index_select(1, idx)[..., -1]
        return x[:, -1:, -1].expand(-1, self.horizon)


class PhysicsReferenceEnsemble(nn.Module):
    """Minimal convex ensemble of three leakage-free PV references.

    The three references are deterministic clear-sky trajectory persistence,
    last-value persistence, and the most recent daily power shape.  Only three
    global logits are learned, so this experiment tests reference quality
    rather than adding another high-capacity router.
    """
    def __init__(self, stats, horizon: int, cycle_len: int):
        super().__init__()
        self.solar = SolarTrajectoryPersistence(stats)
        self.horizon = horizon
        self.cycle_len = cycle_len
        self.reference_logits = nn.Parameter(torch.zeros(3))
        self.last_reference_weights = None
        # Three convex calibration parameters converge far too slowly under
        # the 1e-3 learning rate used for deep baselines.
        self.recommended_lr = 5e-2

    def forward(self, x, cycle=None, future_x=None):
        solar = self.solar(x, cycle, future_x)
        persistence = x[:, -1:, -1].expand(-1, self.horizon)
        if x.shape[1] >= self.cycle_len:
            start = x.shape[1] - self.cycle_len
            idx = start + torch.arange(self.horizon, device=x.device) % self.cycle_len
            daily = x.index_select(1, idx)[..., -1]
        else:
            daily = persistence
        weights = torch.softmax(self.reference_logits, dim=0)
        self.last_reference_weights = weights.detach()
        return (weights[0] * solar + weights[1] * persistence
                + weights[2] * daily)


class ProbabilisticReferenceEnsemble(nn.Module):
    """Reference-ensemble mean with a horizon-wise Gaussian residual scale."""
    def __init__(self, stats, horizon: int, cycle_len: int):
        super().__init__()
        self.base = PhysicsReferenceEnsemble(stats, horizon, cycle_len)
        self.probabilistic_output = True
        self.probabilistic_state = False
        self.uncertainty_log_scale = nn.Parameter(torch.full((horizon,), -1.5))
        self.uncertainty_head = None
        self.register_buffer("uncertainty_calibration", torch.tensor(1.0))
        self.last_predictive_scale = None
        self.recommended_lr = self.base.recommended_lr

    @property
    def reference_logits(self):
        return self.base.reference_logits

    @property
    def last_reference_weights(self):
        return self.base.last_reference_weights

    def forward(self, x, cycle=None, future_x=None):
        pred = self.base(x, cycle, future_x)
        self.last_predictive_scale = (
            F.softplus(self.uncertainty_log_scale).unsqueeze(0).expand_as(pred)
            + 1e-4
        ) * self.uncertainty_calibration
        return pred


def _load_author_module(name: str):
    if str(AMPD_ROOT) not in sys.path:
        sys.path.insert(0, str(AMPD_ROOT))
    # ConvPVNet imports Adaptive_Spectral_Block, although the published
    # AMPDNet class never instantiates it.  Avoid adding a large unused timm
    # dependency while keeping the author model itself unchanged.
    if "TSLANet" not in sys.modules:
        stub = types.ModuleType("TSLANet")
        stub.Adaptive_Spectral_Block = nn.Identity
        sys.modules["TSLANet"] = stub
    return __import__(name)


def _physical_patch_scales(seq_len: int, sample_hours: float,
                           hours=(2.0, 4.0, 8.0)):
    """Convert shared physical-time supports to dataset-specific step counts."""
    scales = []
    for duration in hours:
        patch = min(seq_len, max(2, int(round(duration / max(sample_hours, 1e-6)))))
        stride = max(1, patch // 2)
        if (patch, stride) not in scales:
            scales.append((patch, stride))
    return scales


def _solar_anchor_config(stats):
    clear_name = stats.get("clear_sky_feature")
    future_names = stats.get("solar_future_features") or []
    if clear_name is None or clear_name not in future_names:
        raise ValueError("solar anchor requires a deterministic clear-sky geometry feature")
    clear_idx = int(stats["clear_sky_index"])
    daylight_name = stats.get("daylight_feature")
    cfg = {
        "clear_hist_idx": clear_idx,
        "clear_future_pos": future_names.index(clear_name),
        "target_mu": float(stats["target_mu"]),
        "target_sd": float(stats["target_sd"]),
        "clear_mu": float(stats["feature_mu"][clear_idx]),
        "clear_sd": float(stats["feature_sd"][clear_idx]),
        "clear_max": float(stats["clear_sky_training_max"]),
        "capacity": float(stats["capacity"]),
    }
    if daylight_name in future_names:
        daylight_idx = stats["feature_names"].index(daylight_name)
        cfg.update({
            "daylight_future_pos": future_names.index(daylight_name),
            "daylight_mu": float(stats["feature_mu"][daylight_idx]),
            "daylight_sd": float(stats["feature_sd"][daylight_idx]),
        })
    return cfg


def build_model(name: str, args, channels: int, stats):
    if name == "persistence":
        return Persistence(), "mse"
    if name == "smart_persistence":
        return SolarTrajectoryPersistence(stats), "mse"
    if name == "seasonal_naive":
        return SeasonalNaive(args.horizon, args.cycle_len), "mse"
    if name == "reference_ensemble":
        return PhysicsReferenceEnsemble(stats, args.horizon, args.cycle_len), "mse"
    if name == "reference_ensemble_probglobal":
        return ProbabilisticReferenceEnsemble(
            stats, args.horizon, args.cycle_len,
        ), "mse"
    if name == "ours":
        config = SimpleNamespace(
            seq_len=args.seq_len,
            pred_len=args.horizon,
            enc_in=channels,
            d_model=args.d_model,
            cycle=args.cycle_len,
            sample_hours=stats["dt_hours"],
            physics_indices=stats.get("physics_indices"),
            semantic_prompt_dim=int(stats.get("_dynamic_prompt_dim", 768)),
            patch_hours=tuple(args.ours_patch_hours),
            gtr_period=args.ours_gtr_period,
            revin_mode=args.ours_revin_mode,
            dropout=args.ours_dropout,
            corpatch_heads=args.ours_corpatch_heads,
            cma_heads=args.ours_cma_heads,
            cma_dropout=args.ours_cma_dropout,
        )
        return build_local_model("ours", config), "mse_ramp_phys_night"
    local_names = {
        "dlinear": "dlinear", "patchtst": "patchtst",
        "patchtst_ampd_release": "patchtst", "itransformer": "itransformer",
        "itransformer_ampd_release": "itransformer", "ampdnet": "ampdnet",
        "ampdnet_paper": "ampdnet", "cyclenet": "cyclenet",
        "patchmlp": "patchmlp", "timemixer": "timemixer",
        "timesnet": "timesnet", "crossunet": "crossunet",
    }
    if name in local_names:
        config = SimpleNamespace(
            seq_len=args.seq_len, pred_len=args.horizon, enc_in=channels,
            d_model=args.d_model, patch_len=args.patch_len, stride=args.stride,
            cycle=args.cycle_len,
        )
        model = build_local_model(local_names[name], config)
        if name == "crossunet":
            has_future_nwp = bool(stats.get("future_indices"))
            model.protocol_note = (
                "CrossUNet with audited future NWP"
                if has_future_nwp else
                "CrossUNet-HistoryOnly: released source graph with history fallback; no future NWP"
            )
        return model, "mse"
    if name == "dlinear_legacy":
        mod = _load_author_module("models2")
        return TargetFirst(mod.DLinear(args.seq_len, args.horizon)), "mse"
    if name == "patchtst_legacy":
        mod = _load_author_module("PatchTST")
        net = mod.PatchTST(args.seq_len, channels, args.horizon, args.d_model, 0.1,
                           5, False, 4, 4 * args.d_model, F.gelu, 2,
                           patch_len=args.patch_len, stride=args.stride)
        return TargetFirst(net), "mse"
    if name == "patchtst_ampd_release":
        # Configuration shipped by the AMPDNet authors for their PatchTST
        # comparison: D=128, five encoder layers, P=4, S=2.
        mod = _load_author_module("PatchTST")
        net = mod.PatchTST(
            args.seq_len, channels, args.horizon, 128, 0.01,
            5, False, 4, 256, F.gelu, 5, patch_len=4, stride=2,
        )
        wrapped = TargetFirst(net)
        wrapped.recommended_lr = 1e-3
        wrapped.protocol_note = "AMPDNet-release PatchTST configuration"
        return wrapped, "mse"
    if name == "itransformer_legacy":
        mod = _load_author_module("iTrans_test")
        net = mod.iTransformer(args.seq_len, args.horizon, args.d_model, channels,
                               5, 0.1, False, 4, 4 * args.d_model, F.gelu, 2)
        return TargetFirst(net), "mse"
    if name == "itransformer_ampd_release":
        # Configuration shipped by the AMPDNet authors for their
        # iTransformer comparison: D=128, FFN=256, two encoder layers.
        mod = _load_author_module("iTrans_test")
        net = mod.iTransformer(
            args.seq_len, args.horizon, 128, channels,
            5, 0.01, False, 4, 256, F.gelu, 2,
        )
        wrapped = TargetFirst(net)
        wrapped.recommended_lr = 1e-3
        wrapped.protocol_note = "AMPDNet-release iTransformer configuration"
        return wrapped, "mse"
    if name == "ampdnet_legacy":
        mod = _load_author_module("ConvPVNet")
        net = mod.ConvPVNet(M=channels, L=args.seq_len, T=args.horizon,
                            kernel_sizes=[5, 5, 5, 5], dilat=[1, 2, 3, 4],
                            D=args.d_model, P=args.patch_len, S=args.stride,
                            r=2, dp=0.02, num_layers=2)
        return TargetFirst(net), "mse"
    if name == "ampdnet_paper":
        # Official 2026 Solar Energy setup: D=64, P=4, one AMPD block;
        # stride is 2 for 15-min/1-h and 4 for the 4-h horizon.
        mod = _load_author_module("ConvPVNet")
        paper_stride = 4 if args.horizon >= 16 else 2
        net = mod.ConvPVNet(
            M=channels, L=args.seq_len, T=args.horizon,
            kernel_sizes=[5, 5, 5, 5], dilat=[1, 2, 3, 4],
            D=64, P=4, S=paper_stride, r=2, dp=0.02, num_layers=1,
        )
        return TargetFirst(net), "mse"
    if name == "nwp_transformer":
        future_channels = len(stats.get("future_indices") or [])
        if future_channels <= 0:
            raise ValueError("NWP Transformer requires an audited future feature policy")
        return NWPOnlyTransformer(future_channels, args.horizon, args.d_model), "mse"
    if name == "vanilla_history_nwpdecoder":
        future_channels = len(stats.get("future_indices") or [])
        history_indices = stats.get("core_history_indices") or []
        if future_channels <= 0 or not history_indices:
            raise ValueError("vanilla history decoder requires audited core NWP/history features")
        return VanillaHistoryNWPDecoder(
            history_indices, future_channels, args.seq_len, args.horizon,
            args.d_model, stats["feature_names"],
        ), "mse"

    ours = {
        "pstf": ("tf_parallel_ms", "fact", "mse"),
        "pstf_anchor": ("tf_parallel_ms", "fact", "mse"),
        "pstf_anchor_no_quality": ("tf_parallel_ms", "fact", "mse"),
        "pstf_anchor_static_router": ("tf_parallel_ms_static", "fact", "mse"),
        "pstf_anchor_no_frequency": ("parallel_ms_phys", "fact", "mse"),
        "pstf_anchor_coherence": ("tf_parallel_ms", "coherence", "mse"),
        "pstf_anchor_physloss": ("tf_parallel_ms", "fact", "mse_ramp_phys"),
        "pstf_anchor_routeaux": ("tf_parallel_ms", "fact", "mse"),
        "pstf_anchor_routeaux_physloss": ("tf_parallel_ms", "fact", "mse_ramp_phys"),
        "pstf_anchor_recentstate": ("tf_parallel_ms", "fact", "mse"),
        "pstf_anchor_recentstate_physloss": ("tf_parallel_ms", "fact", "mse_ramp_phys"),
        "pstf_anchor_recentonly": ("tf_parallel_ms", "fact", "mse"),
        "pstf_anchor_recentonly_physloss": ("tf_parallel_ms", "fact", "mse_ramp_phys"),
        "pstf_anchor_static_output": ("tf_parallel_ms", "fact", "mse"),
        "pstf_anchor_no_gtr": ("tf_parallel_ms", "fact", "mse"),
        "pstf_anchor_no_ldrive": ("tf_parallel_ms", "fact", "mse"),
        "pstf_anchor_single_scale": ("tf_parallel_ms", "fact", "mse"),
        "pstf_anchor_nullrouter": ("tf_parallel_ms", "fact", "mse"),
        "pstf_anchor_nullrouter_physloss": ("tf_parallel_ms", "fact", "mse_ramp_phys"),
        "pstf_anchor_nullrouter_physloss_future": ("tf_parallel_ms", "fact", "mse_ramp_phys"),
        "pstf_nullrouter_nwpdecoder_physloss": ("tf_parallel_ms", "fact", "mse_ramp_phys"),
        "pstf_anchor_nullrouter_nwpdecoder_physloss": ("tf_parallel_ms", "fact", "mse_ramp_phys"),
        "pstf_core_nullrouter_nwpdecoder_physloss": ("tf_parallel_ms", "fact", "mse_ramp_phys"),
        # Historical-memory-centred M23 ablations.  The history-only variant
        # is the decisive control for whether NWP is conditioning rather than
        # acting as an independent forecasting branch.
        "pstf_core_nullrouter_physloss": ("tf_parallel_ms", "fact", "mse_ramp_phys"),
        "pstf_core_no_frequency_nwpdecoder_physloss": ("parallel_ms_phys", "fact", "mse_ramp_phys"),
        "pstf_core_no_gtr_nwpdecoder_physloss": ("tf_parallel_ms", "fact", "mse_ramp_phys"),
        "pstf_core_no_ldrive_nwpdecoder_physloss": ("tf_parallel_ms", "fact", "mse_ramp_phys"),
        "pstf_core_single_scale_nwpdecoder_physloss": ("tf_parallel_ms", "fact", "mse_ramp_phys"),
        "pstf_core_coherence_nwpdecoder_physloss": ("tf_parallel_ms", "coherence", "mse_ramp_phys"),
        "pstf_physloss": ("tf_parallel_ms", "fact", "mse_ramp_phys"),
        "pstf_no_quality": ("tf_parallel_ms", "fact", "mse"),
        "pstf_static_router": ("tf_parallel_ms_static", "fact", "mse"),
        "pstf_no_frequency": ("parallel_ms_phys", "fact", "mse"),
        "pstf_coherence": ("tf_parallel_ms", "coherence", "mse"),
        # Revised, paper-facing architecture: one unified temporal encoder at
        # the top level, dynamic-variable spectral response, physical-duration
        # patches, permutation-equivariant variable correlation, and a
        # deterministic solar-trajectory persistence anchor.
        "psrc_v2": ("tf_parallel_ms", "variable", "mse_ramp_phys"),
        "psrc_v2_plain_anchor": ("tf_parallel_ms", "variable", "mse_ramp_phys"),
        "psrc_v2_step_patch": ("tf_parallel_ms", "variable", "mse_ramp_phys"),
        "psrc_v2_fact": ("tf_parallel_ms", "fact", "mse_ramp_phys"),
        "psrc_v2_allfreq": ("tf_parallel_ms", "variable", "mse_ramp_phys"),
        "psrc_v2_scales124": ("tf_parallel_ms", "variable", "mse_ramp_phys"),
        "psrc_v2_scales248": ("tf_parallel_ms", "variable", "mse_ramp_phys"),
        # PCATFR keeps the verified leakage-safe/output-anchor design while
        # making the physical state the explicit controller of both expert
        # routing and frequency-bin selection.
        "pcatfr": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcatfr_summary_state": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcatfr_no_state_spectrum": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcatfr_fixed_router": ("tf_parallel_ms_static", "variable", "mse_ramp_phys_night"),
        "pcatfr_routeaux": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcatfr_direct": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcatfr_residual": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_stateaux": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_fixed_router": ("tf_parallel_ms_static", "variable", "mse_ramp_phys_night"),
        "pcarr_no_null": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_bin_attention": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_energy_only": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_summary_state": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_no_diff": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_sparse_router": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_candidate_aux": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v3_ensemble": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v3_frozen_ensemble": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v3_frozen_ensemble_no_gtr": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v3_frozen_ensemble_no_ldrive": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v3_frozen_ensemble_no_frequency": ("parallel_ms_phys", "variable", "mse_ramp_phys_night"),
        "pcarr_v3_frozen_ensemble_single_scale": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v3_frozen_ensemble_fixed_router": ("tf_parallel_ms_static", "variable", "mse_ramp_phys_night"),
        "pcarr_v3_frozen_ensemble_no_null": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v3_frozen_ensemble_bin_attention": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v3_frozen_ensemble_energy_only": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v3_frozen_ensemble_no_diff": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v3_frozen_ensemble_fact": ("tf_parallel_ms", "fact", "mse_ramp_phys_night"),
        "pcarr_v3_frozen_ensemble_coherence": ("tf_parallel_ms", "coherence", "mse_ramp_phys_night"),
        # Publication-facing physical-controller ablations.  These keep the
        # same numerical backbone and change one causal interface at a time.
        "pcarr_v3_frozen_ensemble_static_output": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v3_frozen_ensemble_energy_only_all_static_output": ("tf_parallel_ms_static", "variable", "mse_ramp_phys_night"),
        "pcarr_v3_frozen_ensemble_no_reference_ensemble": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v3_frozen_ensemble_direct_output": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v3_frozen_ensemble_no_physloss": ("tf_parallel_ms", "variable", "mse_ramp"),
        "pcarr_v3_daily": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v4_prompt": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v4_prompt_shuffle": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v4_stats_mlp": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v5_prompt_adapter": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v5_prompt_sched": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v5_prompt_align": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v6_prompt_tune": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v7_fullprompt_tune": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v8_fullprompt_hybrid": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v8_fullprompt_shuffle": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v8_fullprompt_zero": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v9_promptonly_hybrid": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v9_promptonly_shuffle": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v9_promptonly_zero": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v10_meanlast_hybrid": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v10_meanlast_shuffle": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v10_meanlast_zero": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v11_aligned_hybrid": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v11_aligned_shuffle": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v11_aligned_zero": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v12_zerosafe_hybrid": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v12_zerosafe_shuffle": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v12_zerosafe_zero": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v3_frozen_ensemble_edal": ("tf_parallel_ms", "variable", "edal_ramp_phys_night"),
        "pcarr_v13_t3align": ("tf_parallel_ms", "variable", "edal_ramp_phys_night"),
        "pcarr_v13_t3align_shuffle": ("tf_parallel_ms", "variable", "edal_ramp_phys_night"),
        "pcarr_v13_t3align_zero": ("tf_parallel_ms", "variable", "edal_ramp_phys_night"),
        "pcarr_v13_t3align_random": ("tf_parallel_ms", "variable", "edal_ramp_phys_night"),
        "pcarr_v13_t3align_mse": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v13_t3align_mse_shuffle": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v13_t3align_mse_zero": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v13_t3align_mse_random": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        # V14 replaces redundant window-to-text prompts with frozen channel
        # metadata.  One adapter contributes a zero-safe residual after
        # time-frequency fusion and a relation bias inside CorPatch.
        "pcarr_v14_fsra": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v14_fsra_shuffle": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v14_fsra_zero": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v14_fsra_random": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v14_fsra_id": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v14_fsra_relationonly": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v14_fsra_residualonly": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v14_fsra_roles": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v14_fsra_roles_relationonly": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v14_fsra_roles_residualonly": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v14_fsra_roles_adaptive3_residualonly": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v14_fsra_roles_adaptive3_paired_residualonly": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v14_fsra_roles_adaptive3_paired_probglobal_residualonly": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v14_fsra_roles_adaptive3_paired_probstate_residualonly": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v14_fsra_roles_adaptive3_paired_probquantile_residualonly": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v14_fsra_roles_adaptive3_paired_mcdropout_residualonly": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v14_fsra_roles_adaptive3_paired_decomp_residualonly": ("tf_parallel_ms", "variable", "mse_ramp_phys_night_decomp"),
        "pcarr_v14_fsra_roles_adaptive3_paired_decomp_probglobal_residualonly": ("tf_parallel_ms", "variable", "mse_ramp_phys_night_decomp"),
        "pcarr_v14_fsra_roles_adaptive3_paired_decomp_probstate_residualonly": ("tf_parallel_ms", "variable", "mse_ramp_phys_night_decomp"),
        "pcarr_v14_fsra_roles_adaptive3_transfer_residualonly": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v14_fsra_roles_adaptive3_transfergrid_residualonly": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v14_fsra_roles_adaptive3_transfergrid_paired_residualonly": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        # V15 restores a genuine frozen GPT-2 modality, but aligns three causal
        # history prompts only with their matching 2/4/8 h patch scales.  The
        # verified numerical forecaster remains the exact strength-zero path.
        "pcarr_v15_gpt2_ms": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v15_gpt2_ms_shuffle": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v15_gpt2_ms_zero": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v15_gpt2_ms_random": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v15_gpt2_ms_patchonly": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        # V16 is the low-risk composition: retain the verified v14 role-aware
        # channel adapter, then fit the zero-safe multi-scale GPT-2 prompt
        # adapter on top while the complete v14 forecaster stays frozen.
        "pcarr_v16_fsra_roles_adaptive3_paired_residualonly_gpt2ms": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        # Paired architecture check: keep the complete v16 graph unchanged,
        # but feed the selected raw dynamic channels directly to rFFT instead
        # of pre-emphasising them with a first difference.  The ``no_diff``
        # suffix is consumed by ``spectral_disturbance`` below.
        "pcarr_v16_fsra_roles_adaptive3_paired_residualonly_gpt2ms_no_diff": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        # Paper-facing simplified spectral expert: raw dynamic sequence ->
        # rFFT -> Spectral-MKAN -> irFFT.  Unlike the diagnostic ``no_diff``
        # variant above, this removes both physical-band energy routing and
        # state-conditioned per-bin attention; physical state acts only at the
        # outer temporal/frequency/identity residual router.
        "pcarr_v16_fsra_roles_adaptive3_paired_residualonly_gpt2ms_rawsimple_no_diff": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v16_fsra_roles_adaptive3_paired_residualonly_gpt2ms_promptshuffle": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v16_fsra_roles_adaptive3_paired_residualonly_gpt2ms_promptrandom": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v16_fsra_roles_adaptive3_paired_residualonly_gpt2ms_statsmlp": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v17_fsra_roles_adaptive3_paired_residualonly_gpt2scale": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v17_fsra_roles_adaptive3_paired_residualonly_gpt2scale_promptshuffle": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v17_fsra_roles_adaptive3_paired_residualonly_gpt2scale_promptrandom": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v17_fsra_roles_adaptive3_paired_residualonly_gpt2scale_statsmlp": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v18_fsra_roles_adaptive3_paired_residualonly_qwen": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v18_fsra_roles_adaptive3_paired_residualonly_qwen_promptshuffle": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v18_fsra_roles_adaptive3_paired_residualonly_qwen_promptrandom": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v18_fsra_roles_adaptive3_paired_residualonly_qwen_statsmlp": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v19_fsra_roles_adaptive3_paired_residualonly_qwen_tfrouter": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v19_fsra_roles_adaptive3_paired_residualonly_qwen_tfrouter_promptshuffle": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v19_fsra_roles_adaptive3_paired_residualonly_qwen_tfrouter_promptrandom": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v19_fsra_roles_adaptive3_paired_residualonly_qwen_tfrouter_statsmlp": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        # V20 makes the deterministic physical reference a rejectable
        # history-conditioned residual correction rather than the primary
        # forecast. V21 adds frozen GPT-2 operating-regime prototypes only as
        # priors on the existing TF and scale router logits.
        "pcarr_v20_refrel": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v21_fsra_roles_adaptive3_paired_residualonly_refrel": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v21_fsra_roles_adaptive3_paired_residualonly_refrel_fsra_shuffle": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v21_fsra_roles_adaptive3_paired_residualonly_refrel_fsra_random": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v21_fsra_roles_adaptive3_paired_residualonly_refrel_fsra_zero": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v21_fsra_roles_adaptive3_paired_residualonly_refrel_gpt2proto": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v21_fsra_roles_adaptive3_paired_residualonly_refrel_gpt2proto_promptshuffle": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v21_fsra_roles_adaptive3_paired_residualonly_refrel_gpt2proto_promptrandom": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v21_fsra_roles_adaptive3_paired_residualonly_refrel_gpt2proto_promptzero": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v22_fsra_roles_adaptive3_paired_residualonly_refrel_gpt2semantic_a": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v22_fsra_roles_adaptive3_paired_residualonly_refrel_gpt2semantic_b": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v22_fsra_roles_adaptive3_paired_residualonly_refrel_gpt2semantic_c": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        # V23 follows T3Time's effective fusion path without changing the
        # established PV backbone: ordered per-variable history prompts are
        # encoded by frozen GPT-2, adapted by a trainable prompt encoder,
        # aligned to fused time-frequency features by four independent CMA
        # heads, and injected through a conservative channel residual.
        "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma_spectralfix_off": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma_spectralfix_full": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v23_fsra_roles_adaptive3_paired_probstate_residualonly_refrel_t3cma": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v23_fsra_roles_adaptive3_paired_probquantile_residualonly_refrel_t3cma": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v23_fsra_roles_adaptive3_paired_mcdropout_residualonly_refrel_t3cma": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma_fixed_router": ("tf_parallel_ms_static", "variable", "mse_ramp_phys_night"),
        "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma_single_scale": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma_no_frequency": ("parallel_ms_phys", "variable", "mse_ramp_phys_night"),
        "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma_spectral_zero": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma_clean_spectral_off": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma_no_gtr": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma_no_ldrive": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma_promptzero": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma_zero": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        # Frequency-v2 validation controls.  Every variant instantiates the
        # same dual-band experts; a fixed branch mask is the sole difference.
        "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma_freqv2_temporal": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma_freqv2_raw_low": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma_freqv2_detrended_mid_high": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma_freqv2_dual": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        # Frequency-v3 controls: the input-side spectrum is removed for every
        # group and an endogenous-target complex gate is placed immediately
        # before the shared forecast head.  Loss and gate factors are crossed
        # while parameter allocation remains identical.
        "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma_freqv3_temporal": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma_freqv3_gate": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma_freqv3_loss": ("tf_parallel_ms", "variable", "mse_ramp_phys_night_decomp"),
        "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma_freqv3_gate_loss": ("tf_parallel_ms", "variable", "mse_ramp_phys_night_decomp"),
        "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma_no_gtr_freqv3_gate_loss": ("tf_parallel_ms", "variable", "mse_ramp_phys_night_decomp"),
        "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma_no_ldrive_freqv3_gate_loss": ("tf_parallel_ms", "variable", "mse_ramp_phys_night_decomp"),
        "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma_fixed_router_freqv3_gate_loss": ("tf_parallel_ms_static", "variable", "mse_ramp_phys_night_decomp"),
        "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma_single_scale_freqv3_gate_loss": ("tf_parallel_ms", "variable", "mse_ramp_phys_night_decomp"),
        # Frequency-v4 keeps the original pre-Patch spectral insertion point.
        # Only the internal target-led global/local spectral expert changes.
        "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma_freqv4_off": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma_freqv4_global_low": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma_freqv4_local_cloud": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma_freqv4_dual": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        # Frequency-v5 changes only the original parallel spectral expert.
        # It can abstain through an internal null route and uses bounded,
        # channel-reliable residuals.  Off/adaptive are parameter matched.
        "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma_freqv5_off": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma_freqv5_adaptive": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        # Frequency-v6 preserves the original parallel insertion point while
        # changing only the internal expert to target-only hierarchical
        # spectral calibration.  Off/full are parameter matched.
        "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma_freqv6_off": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma_freqv6_full": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        # Frequency-v7 keeps the exact original parallel branch and fusion
        # point.  Only its target-led local spectral expert changes; off/full
        # use the same outer TF softmax without reallocation.
        "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma_freqv7_off": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma_freqv7_full": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        # Frequency-v8 restores the original full-window multivariate complex
        # spectrum while robustly limiting its inputs and residual internally.
        "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma_freqv8_off": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma_freqv8_full": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma_freqv9_off": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma_freqv9_full": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma_no_gtr_freqv9_full": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma_no_ldrive_freqv9_full": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma_fixed_router_freqv9_full": ("tf_parallel_ms_static", "variable", "mse_ramp_phys_night"),
        "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma_single_scale_freqv9_full": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
        "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma_fsrazero_freqv9_full": ("tf_parallel_ms", "variable", "mse_ramp_phys_night"),
    }
    if name not in ours:
        raise ValueError(f"unknown model {name}")
    mode, encoder, loss_kind = ours[name]
    revised_v2 = name.startswith("psrc_v2")
    pcatfr = name.startswith("pcatfr")
    pcarr = name.startswith("pcarr")
    fsra_model = (name.startswith("pcarr_v14_fsra")
                  or name.startswith("pcarr_v16_fsra")
                  or name.startswith("pcarr_v17_fsra")
                  or name.startswith("pcarr_v18_fsra")
                  or name.startswith("pcarr_v19_fsra")
                  or name.startswith("pcarr_v21_fsra")
                  or name.startswith("pcarr_v22_fsra")
                  or name.startswith("pcarr_v23_fsra"))
    scale_prompt_model = (name.startswith("pcarr_v15_gpt2_ms")
                          or name.startswith("pcarr_v16_fsra")
                          or name.startswith("pcarr_v17_fsra")
                          or name.startswith("pcarr_v18_fsra")
                          or name.startswith("pcarr_v19_fsra")
                          or "v21_fsra" in name and "gpt2proto" in name
                          or "v22_fsra" in name and "gpt2semantic" in name)
    reference_residual_model = (
        name.startswith("pcarr_v20_refrel")
        or name.startswith("pcarr_v21_fsra")
        or name.startswith("pcarr_v22_fsra")
        or name.startswith("pcarr_v23_fsra")
    )
    paper_model = revised_v2 or pcatfr or pcarr
    core_model = name.startswith("pstf_core_")
    if core_model:
        selected = stats.get("core_history_indices") or []
        if not selected:
            raise ValueError("core model requires GEFCom core history indices")
        original_physics = set(stats.get("physics_indices") or [])
        original_quality = set(stats.get("quality_indices") or [])
        net_channels = len(selected)
        physics_indices = [j for j, i in enumerate(selected) if i in original_physics]
        remapped_quality = [j for j, i in enumerate(selected) if i in original_quality]
    else:
        selected = None
        net_channels = channels
        physics_indices = stats.get("physics_indices")
        remapped_quality = stats.get("quality_indices")
    quality_indices = [] if "no_quality" in name else remapped_quality
    nwp_decoder = "nwpdecoder" in name
    if nwp_decoder and not stats.get("future_indices"):
        raise ValueError(
            "NWP-conditioned decoder requires --future-feature-policy gefcom_nwp"
        )
    legacy_direct_output = (name.endswith("direct") or "direct_output" in name)
    legacy_residual_output = name.endswith("residual")
    anchor_output_mode = args.anchor_output_mode
    if anchor_output_mode == "gated":
        if legacy_direct_output:
            anchor_output_mode = "direct"
        elif legacy_residual_output:
            anchor_output_mode = "residual"
    frequency_v2_mode = None
    for suffix, mode_name in (
        ("_freqv2_temporal", "temporal"),
        ("_freqv2_raw_low", "raw_low"),
        ("_freqv2_detrended_mid_high", "detrended_mid_high"),
        ("_freqv2_dual", "dual"),
    ):
        if name.endswith(suffix):
            frequency_v2_mode = mode_name
            break
    frequency_v3_mode = None
    for suffix, mode_name in (
        ("_freqv3_gate_loss", "gate"),
        ("_freqv3_temporal", "off"),
        ("_freqv3_gate", "gate"),
        ("_freqv3_loss", "off"),
    ):
        if name.endswith(suffix):
            frequency_v3_mode = mode_name
            break
    frequency_v4_mode = None
    for suffix, mode_name in (
        ("_freqv4_off", "off"),
        ("_freqv4_global_low", "global_low"),
        ("_freqv4_local_cloud", "local_cloud"),
        ("_freqv4_dual", "dual"),
    ):
        if name.endswith(suffix):
            frequency_v4_mode = mode_name
            break
    frequency_v5_mode = None
    for suffix, mode_name in (
        ("_freqv5_off", "off"),
        ("_freqv5_adaptive", "adaptive"),
    ):
        if name.endswith(suffix):
            frequency_v5_mode = mode_name
            break
    frequency_v6_mode = None
    for suffix, mode_name in (
        ("_freqv6_off", "off"),
        ("_freqv6_full", "full"),
    ):
        if name.endswith(suffix):
            frequency_v6_mode = mode_name
            break
    frequency_v7_mode = None
    for suffix, mode_name in (
        ("_freqv7_off", "off"),
        ("_freqv7_full", "full"),
    ):
        if name.endswith(suffix):
            frequency_v7_mode = mode_name
            break
    frequency_v8_mode = None
    for suffix, mode_name in (
        ("_freqv8_off", "off"),
        ("_freqv8_full", "full"),
    ):
        if name.endswith(suffix):
            frequency_v8_mode = mode_name
            break
    frequency_v9_mode = None
    for suffix, mode_name in (
        ("_freqv9_off", "off"),
        ("_freqv9_full", "full"),
    ):
        if name.endswith(suffix):
            frequency_v9_mode = mode_name
            break
    spectral_mkan_mode = (
        "residual_off" if name.endswith("_spectralfix_off") else
        "residual_full" if name.endswith("_spectralfix_full") else
        "legacy"
    )
    if paper_model:
        if frequency_v8_mode is not None or frequency_v9_mode is not None:
            # The actual DKASC spectral audit found that raw wind direction,
            # wind speed and an outlying diffuse-tilted sensor contributed
            # 92% of differentiated spectral energy despite negligible PV
            # coherence.  Keep the original multivariate mechanism, but feed
            # only the two reliable irradiance responses plus PV Target.
            allowed = set(stats.get("spectral_indices") or [])
            core_names = {
                "target", "global_horizontal_radiation",
                "radiation_global_tilted", "ghi", "gti",
            }
            frequency_indices = [
                index for index, feature_name in enumerate(stats["feature_names"])
                if index in allowed and feature_name.lower() in core_names
            ]
            if len(frequency_indices) < 2 or not any(
                    stats["feature_names"][i].lower() == "target"
                    for i in frequency_indices):
                raise ValueError(
                    "selected frequency expert requires Target and at least one reliable irradiance channel"
                )
        elif (frequency_v2_mode is not None or frequency_v4_mode is not None
                or frequency_v6_mode is not None
                or frequency_v7_mode is not None):
            allowed = set(stats.get("spectral_indices") or [])
            frequency_indices = []
            for index, feature_name in enumerate(stats["feature_names"]):
                normalized_name = feature_name.lower()
                physically_coupled = (
                    normalized_name == "target"
                    or "radiation" in normalized_name
                    or "irradiance" in normalized_name
                    or "ghi" in normalized_name
                    or "dhi" in normalized_name
                )
                if index in allowed and physically_coupled:
                    frequency_indices.append(index)
            if not frequency_indices:
                raise ValueError(
                    "selected frequency expert requires PV/irradiance history channels"
                )
        else:
            frequency_indices = (list(range(net_channels)) if name.endswith("allfreq")
                                 else stats.get("spectral_indices"))
        if name.endswith("step_patch"):
            physical_scales = None
        elif name.endswith("scales124"):
            physical_scales = _physical_patch_scales(
                args.seq_len, stats["dt_hours"], hours=(1.0, 2.0, 4.0))
        elif name.endswith("scales248"):
            physical_scales = _physical_patch_scales(
                args.seq_len, stats["dt_hours"], hours=(2.0, 4.0, 8.0))
        else:
            physical_scales = _physical_patch_scales(args.seq_len, stats["dt_hours"])
        solar_cfg = (None if (name.endswith("plain_anchor")
                              or anchor_output_mode == "direct")
                     else _solar_anchor_config(stats))
    else:
        frequency_indices = None
        physical_scales = None
        solar_cfg = None
    if name == "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma":
        config = SimpleNamespace(
            seq_len=args.seq_len, pred_len=args.horizon, enc_in=net_channels,
            d_model=args.d_model, patch_len=args.patch_len, stride=args.stride,
            cycle=args.cycle_len, sample_hours=stats["dt_hours"],
            physics_indices=physics_indices, quality_indices=quality_indices,
            semantic_prompt_dim=int(stats.get("_dynamic_prompt_dim", 768)),
            channel_semantic_embeddings=stats.get("_fsra_vectors"),
            # Main comparison is history-only.  The deterministic solar anchor
            # is an explicit validation ablation rather than a hidden advantage.
            solar_anchor_config=solar_cfg if args.ours_solar_anchor else None,
            patch_hours=tuple(args.ours_patch_hours),
            gtr_period=args.ours_gtr_period,
            revin_mode=args.ours_revin_mode,
            dropout=args.ours_dropout,
            corpatch_heads=args.ours_corpatch_heads,
            cma_heads=args.ours_cma_heads,
            cma_dropout=args.ours_cma_dropout,
            cma_scale_route=args.ours_cma_scale_route,
            cma_prompt_encoder=(
                "numeric_mlp"
                if args.ours_cma_prompt_features == "numeric_sequence"
                else "transformer"
            ),
        )
        return LegacyOursModel(config), loss_kind
    net = AblationRPGTR(args.seq_len, args.horizon, net_channels, args.cycle_len,
                args.patch_len, args.stride, args.d_model, encoder, mode,
                revin=True, physics_indices=physics_indices,
                quality_indices=quality_indices,
                output_anchor=(("anchor" in name) or paper_model)
                and anchor_output_mode == "gated"
                and not reference_residual_model,
                recent_physics=("recentstate" in name),
                recent_only_physics=("recentonly" in name),
                single_scale=("single_scale" in name),
                 disable_gtr=("no_gtr" in name),
                 disable_ldrive=("no_ldrive" in name),
                 disable_frequency=("spectral_zero" in name),
                 clean_frequency_ablation=("clean_spectral_off" in name),
                static_output_anchor=("static_output" in name),
                null_route=(("nullrouter" in name) or paper_model) and not name.endswith("no_null"),
                future_covariates=("future" in name),
                nwp_conditioned_decoder=nwp_decoder,
                future_channels=len(stats.get("future_indices") or []),
                frequency_indices=frequency_indices,
                physical_scales=physical_scales,
                permutation_correlation=paper_model,
                solar_anchor_config=solar_cfg,
                reference_ensemble=(("v3_ensemble" in name or "v3_frozen_ensemble" in name
                                    or "v4_" in name or "v5_" in name or "v6_" in name
                                    or "v7_" in name or "v8_" in name or "v9_" in name
                                    or "v10_" in name or "v11_" in name or "v12_" in name
                                    or "v13_" in name or "v14_" in name or "v15_" in name
                                    or "v16_" in name or "v17_" in name or "v18_" in name
                                    or "v19_" in name or "v20_" in name or "v21_" in name
                                    or "v22_" in name or "v23_" in name)
                                    and "no_reference_ensemble" not in name
                                    and anchor_output_mode != "direct"),
                daily_shape_anchor=name.endswith("v3_daily"),
                sequential_physics=((pcatfr or pcarr) and not name.endswith("summary_state")),
                state_conditioned_spectrum=(
                    ((pcatfr and not name.endswith("no_state_spectrum")) or pcarr)
                    and "rawsimple" not in name
                ),
                solar_residual_output=(anchor_output_mode == "residual"),
                physical_band_spectrum=(
                    pcarr and not name.endswith("bin_attention")
                    and "rawsimple" not in name
                ),
                sample_hours=stats["dt_hours"],
                state_auxiliary=name.endswith("stateaux"),
                state_conditioned_bands=("energy_only" not in name),
                spectral_disturbance=((not name.endswith("no_diff")) if pcarr else None),
                semantic_prompt_dim=(1536 if ("v10_meanlast" in name or "v11_aligned" in name) else
                                     768 if ("v4_prompt" in name or "v5_prompt" in name
                                              or "v6_prompt" in name or "v7_fullprompt" in name
                                              or "v8_fullprompt" in name or "v9_promptonly" in name) else
                                     1536 if "v13_t3align" in name else
                                     768 if "v23_fsra" in name else
                                     768 if "v12_zerosafe" in name else
                                     2 if name == "pcarr_v4_stats_mlp" else 0),
                adaptive_semantic_alignment=("v5_prompt" in name or "v6_prompt" in name
                                             or "v7_fullprompt" in name or "v8_fullprompt" in name
                                             or "v9_promptonly" in name or "v10_meanlast" in name
                                             or "v11_aligned" in name or "v12_zerosafe" in name
                                             or "v13_t3align" in name or "v23_fsra" in name),
                semantic_output_calibration=("v6_prompt" in name or "v7_fullprompt" in name
                                             or "v8_fullprompt" in name or "v9_promptonly" in name
                                             or "v10_meanlast" in name or "v11_aligned" in name
                                             or "v12_zerosafe" in name),
                prompt_only_semantic_output=("v9_promptonly" in name or "v10_meanlast" in name
                                             or "v11_aligned" in name or "v12_zerosafe" in name),
                zero_safe_semantic=("v12_zerosafe" in name or "v13_t3align" in name
                                    or "v23_fsra" in name),
                semantic_cma_heads=(4 if "v23_fsra" in name else
                                    3 if "v13_t3align" in name else 0),
                semantic_contrastive_alignment=not (
                    "v13_t3align" in name or "v23_fsra" in name
                ),
                semantic_prompt_encoder_layers=(1 if "v23_fsra" in name else 0),
                semantic_channel_residual=("v23_fsra" in name),
                channel_semantic_embeddings=stats.get("_fsra_vectors"),
                channel_semantic_trainable=name.endswith("_id"),
                fsra_residual=(fsra_model and "relationonly" not in name),
                fsra_relation_bias=(fsra_model and "residualonly" not in name),
                fsra_channel_alignment=fsra_model,
                fsra_cma_heads=(3 if "adaptive3" in name else 0),
                fsra_transferable=("transfer" in name),
                fsra_transfer_bins=(24 if "transfergrid" in name else 0),
                fsra_paired_alignment=("paired" in name),
                probabilistic_output=("probglobal" in name or "probstate" in name),
                probabilistic_state=("probstate" in name),
                quantile_output=("probquantile" in name),
                quantile_state=("probquantile" in name),
                output_correction_floor=args.output_correction_floor,
                output_router_bias=args.output_router_bias,
                local_token_refiner=args.local_token_refiner,
                local_refiner_dropout=args.local_refiner_dropout,
                scale_token_mixer=args.scale_token_mixer,
                scale_token_mixer_adaptive=not args.fixed_scale_token_mixer,
                scale_token_mixer_dropout=args.scale_token_mixer_dropout,
                # V16 attaches this module only after the exact v14+FSRA
                # checkpoint is finished; pre-allocation changed low-level GPU
                # training trajectories despite equal seeds and weights.
                scale_prompt_dim=(1536 if name.startswith("pcarr_v15_gpt2_ms") else 0),
                 scale_prompt_heads=(3 if name.startswith("pcarr_v15_gpt2_ms") else 1),
                 scale_prompt_output_gate=not name.endswith("patchonly"),
                 # A direct-output control must not instantiate, require, or
                 # consume the deterministic future solar reference merely
                 # because the publication model name contains ``refrel``.
                 reference_residual_gate=(
                     reference_residual_model and anchor_output_mode != "direct"
                 ),
                 frequency_v2_mode=frequency_v2_mode,
                 frequency_v3_mode=frequency_v3_mode,
                 frequency_v4_mode=frequency_v4_mode,
                 frequency_v5_mode=frequency_v5_mode,
                 frequency_v6_mode=frequency_v6_mode,
                 frequency_v7_mode=frequency_v7_mode,
                 frequency_v8_mode=frequency_v8_mode,
                 frequency_v9_mode=frequency_v9_mode,
                 spectral_mkan_mode=spectral_mkan_mode,
                 router_activation=args.router_activation,
                expert_drop_path=args.expert_drop_path,
                gtr_dropout=args.gtr_dropout)
    if core_model:
        net = HistorySelectWrapper(net, selected, stats["feature_names"])
    return net, loss_kind


def _masked_mean(value, mask):
    mask = mask.to(value.dtype)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def _average_state_dicts(states):
    """Average floating tensors while preserving discrete buffers.

    States must already be ordered from best to worst validation score.  The
    best checkpoint supplies integer/counter buffers; trainable parameters and
    floating buffers are averaged in float64 and cast back to their original
    dtype to minimize accumulation error.
    """
    if not states:
        raise ValueError("checkpoint averaging requires at least one state")
    averaged = {}
    for key, reference in states[0].items():
        if reference.is_floating_point() or reference.is_complex():
            work_dtype = torch.complex128 if reference.is_complex() else torch.float64
            value = torch.stack([state[key].to(work_dtype) for state in states]).mean(dim=0)
            averaged[key] = value.to(reference.dtype)
        else:
            averaged[key] = reference.clone()
    return averaged


def _training_point_mask(y, mask, stats, daylight_weight=0.0):
    """Return a mean-one training mask with optional daylight emphasis.

    Daylight is derived from the *training target* (power above 1% of the
    training-split capacity).  It is therefore supervision, not a model input,
    and cannot expose validation/test targets.  Renormalizing the weighted mask
    keeps the effective loss scale stable while the daylight weight is tuned.
    """
    observed = torch.ones_like(y) if mask is None else mask
    weight = float(daylight_weight)
    if weight <= 0.0:
        return observed
    target_physical = y * stats["target_sd"] + stats["target_mu"]
    threshold = 0.01 * max(float(stats["capacity"]), 1e-6)
    daylight = (target_physical > threshold).to(y.dtype)
    weighted = observed * (1.0 + weight * daylight)
    return weighted * (
        observed.sum().clamp_min(1.0) / weighted.sum().clamp_min(1.0)
    )


def _selection_score(metric, mode, reference=None):
    """Validation-only checkpoint score used by every deterministic stage."""
    if mode == "all_mse":
        return float(metric["mse_standardized"])
    if mode == "daylight_mse":
        return float(metric["daylight_mse_standardized"])
    if mode != "daylight_combo":
        raise ValueError(f"Unknown selection metric: {mode}")
    mse = float(metric["daylight_mse_standardized"])
    mae = float(metric["daylight_mae_standardized"])
    if reference is None:
        return mse + mae
    return (
        mse / max(float(reference["daylight_mse_standardized"]), 1e-8)
        + mae / max(float(reference["daylight_mae_standardized"]), 1e-8)
    )


def _gaussian_crps_torch(pred, target, scale):
    """Closed-form CRPS for N(pred, scale^2), in target-standardized units."""
    scale = scale.clamp_min(1e-6)
    z = (target - pred) / scale
    phi = torch.exp(-0.5 * z.square()) / np.sqrt(2.0 * np.pi)
    cdf = 0.5 * (1.0 + torch.erf(z / np.sqrt(2.0)))
    return scale * (
        z * (2.0 * cdf - 1.0) + 2.0 * phi - 1.0 / np.sqrt(np.pi)
    )


@torch.no_grad()
def _validation_conformal_scale(model, loader, device):
    """Return one validation-only multiplier for marginal 90% coverage."""
    model.eval()
    model.uncertainty_calibration.fill_(1.0)
    scores = []
    for batch in loader:
        x, y, y_mask, cycle, future_x, prompt_x = unpack_batch(batch, device)
        pred = predict(model, x, cycle, future_x, prompt_x)
        scale = model.last_predictive_scale.clamp_min(1e-6)
        observed = y_mask > 0.5
        scores.append((torch.abs(y - pred) / scale)[observed].cpu())
    if not scores:
        return 1.0
    values = torch.cat(scores).numpy()
    # Higher empirical quantile is conservative on finite validation sets.
    try:
        quantile = float(np.quantile(values, 0.90, method="higher"))
    except TypeError:  # NumPy < 1.22
        quantile = float(np.quantile(values, 0.90, interpolation="higher"))
    factor = quantile / 1.6448536269514722
    return float(np.clip(factor, 0.25, 4.0))


@torch.no_grad()
def _validation_quantile_adjustment(model, loader, device):
    """One-sided split-conformal widening for a learned 5--95% interval."""
    model.eval()
    model.quantile_interval_adjustment.fill_(0.0)
    scores = []
    for batch in loader:
        x, y, y_mask, cycle, future_x, prompt_x = unpack_batch(batch, device)
        predict(model, x, cycle, future_x, prompt_x)
        members = model.last_predictive_members
        lower, upper = members[..., 0], members[..., -1]
        observed = y_mask > 0.5
        score = torch.maximum(lower - y, y - upper)
        scores.append(score[observed].cpu())
    if not scores:
        return 0.0
    values = torch.cat(scores).numpy()
    try:
        adjustment = float(np.quantile(values, 0.90, method="higher"))
    except TypeError:
        adjustment = float(np.quantile(values, 0.90, interpolation="higher"))
    # Do not use validation data to sharpen an already conservative interval.
    return float(np.clip(adjustment, 0.0, 4.0))


def physical_loss(pred, y, kind, stats, mask=None, future_x=None,
                  daylight_weight=0.0, ramp_weight=0.05):
    mask = torch.ones_like(y) if mask is None else mask
    point_mask = _training_point_mask(
        y, mask, stats, daylight_weight=daylight_weight,
    )
    if kind.startswith("mse_mae"):
        squared = _masked_mean((pred - y).square(), point_mask)
        absolute = _masked_mean((pred - y).abs(), point_mask)
        loss = 0.5 * (squared + absolute)
    elif kind.startswith("mae"):
        loss = _masked_mean((pred - y).abs(), point_mask)
    elif "edal" in kind:
        # Energy-aware DAL: retain the robust rational-quadratic/logarithmic
        # core from the supplied loss study, but use only one non-redundant
        # event weight.  The target-derived ramp is available during training
        # only and is never an input to the forecaster.  Normalizing weights to
        # mean one keeps the optimization scale comparable across sites.
        error2 = (pred - y).square()
        robust_point = 0.5 * error2 / (error2 + 0.5) + 0.5 * torch.log1p(error2 / 0.5)
        p_target = y * stats["target_sd"] + stats["target_mu"]
        cap = max(float(stats["capacity"]), 1e-6)
        if y.shape[1] > 1:
            target_ramp = torch.abs(torch.diff(p_target, dim=1)) / (cap * stats["dt_hours"])
            target_ramp = F.pad(target_ramp, (1, 0), mode="replicate")
            event_weight = 1.0 + torch.clamp(target_ramp / 0.10, 0.0, 2.0)
        else:
            event_weight = torch.ones_like(y)
        event_weight = event_weight / _masked_mean(event_weight, point_mask).clamp_min(1e-6)
        loss = _masked_mean(event_weight * robust_point, point_mask)
    else:
        loss = _masked_mean((pred - y).square(), point_mask)
    if "ramp" in kind and pred.shape[1] > 1:
        p = pred * stats["target_sd"] + stats["target_mu"]
        t = y * stats["target_sd"] + stats["target_mu"]
        cap = max(stats["capacity"], 1e-6)
        ramp_mask = mask[:, 1:] * mask[:, :-1]
        ramp_error = torch.abs(torch.diff(p, dim=1) - torch.diff(t, dim=1))
        loss = loss + float(ramp_weight) * _masked_mean(
            ramp_error, ramp_mask,
        ) / (cap * stats["dt_hours"])
    if "phys" in kind:
        p = pred * stats["target_sd"] + stats["target_mu"]
        t = y * stats["target_sd"] + stats["target_mu"]
        cap = max(stats["capacity"], 1e-6)
        loss = loss + 0.02 * (F.relu(-p / cap).mean() + F.relu((p - cap) / cap).mean())
        p_mean = (p * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        t_mean = (t * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        valid_window = (mask.sum(1) > 0).to(p.dtype)
        loss = loss + 0.02 * _masked_mean((p_mean - t_mean).abs(), valid_window) / cap
    if "night" in kind and future_x is not None:
        future_names = stats.get("solar_future_features") or []
        daylight_name = stats.get("daylight_feature")
        clear_name = stats.get("clear_sky_feature")
        if daylight_name in future_names:
            source_name = daylight_name
            source_idx = stats["feature_names"].index(source_name)
            source = (future_x[..., future_names.index(source_name)]
                      * stats["feature_sd"][source_idx] + stats["feature_mu"][source_idx])
            night = source <= 0.5
        elif clear_name in future_names:
            source_idx = stats["feature_names"].index(clear_name)
            source = (future_x[..., future_names.index(clear_name)]
                      * stats["feature_sd"][source_idx] + stats["feature_mu"][source_idx])
            night = source <= 0.02 * max(float(stats.get("clear_sky_training_max") or 1.0), 1e-6)
        else:
            night = None
        if night is not None and night.any():
            p = pred * stats["target_sd"] + stats["target_mu"]
            loss = loss + 0.02 * (p[night] / max(stats["capacity"], 1e-6)).square().mean()
    if "decomp" in kind and pred.shape[1] >= 4:
        # CEEMDAN's useful principle is scale-separated supervision, not the
        # non-causal offline decomposition itself.  A fixed physical-frequency
        # loss compares output energy in slow weather evolution (<0.1 cph),
        # cloud-transition (0.1-1 cph), and fast fluctuation (>1 cph) bands.
        # Pointwise MSE still supplies phase/timing supervision.  The Hann
        # window limits boundary leakage and the small coefficient keeps this
        # term auxiliary. Missing-label windows are down-weighted by their
        # genuinely observed fraction.
        horizon = pred.shape[1]
        window = torch.hann_window(
            horizon, periodic=False, device=pred.device, dtype=pred.dtype,
        ).unsqueeze(0)
        pred_centered = pred - pred.mean(dim=1, keepdim=True)
        target_centered = y - y.mean(dim=1, keepdim=True)
        pred_amp = torch.log1p(
            torch.fft.rfft(pred_centered * window, dim=1).abs() / np.sqrt(horizon)
        )
        target_amp = torch.log1p(
            torch.fft.rfft(target_centered * window, dim=1).abs() / np.sqrt(horizon)
        )
        freqs = torch.fft.rfftfreq(
            horizon, d=float(stats["dt_hours"]), device=pred.device,
        )
        band_masks = (
            (freqs > 0.0) & (freqs < 0.1),
            (freqs >= 0.1) & (freqs <= 1.0),
            freqs > 1.0,
        )
        sample_weight = mask.mean(dim=1)
        band_terms = []
        for band_mask in band_masks:
            if band_mask.any():
                per_sample = F.smooth_l1_loss(
                    pred_amp[:, band_mask], target_amp[:, band_mask],
                    reduction="none",
                ).mean(dim=1)
                band_terms.append(
                    (per_sample * sample_weight).sum()
                    / sample_weight.sum().clamp_min(1.0)
                )
        if band_terms:
            loss = loss + 0.02 * torch.stack(band_terms).mean()
    return loss


def physical_routing_loss(model, x, stats):
    """Weakly supervise routing direction from history-only physical states.

    Disturbance combines normalized ramp intensity and clear-sky deviation.
    Spectral routing is additionally discounted when observations are sparse,
    preventing interpolated boundaries from being rewarded as real spectra.
    """
    if model.route_scale_weights is None or model.route_tf_weights is None:
        return x.new_zeros(())
    cap = max(float(stats["capacity"]), 1e-6)
    target = x[..., -1] * stats["target_sd"] + stats["target_mu"]
    dx = torch.diff(target, dim=1)
    ramp = dx.abs().mean(1) / (cap * stats["dt_hours"] + 1e-6)
    disturbance = ramp
    names = stats["feature_names"]
    clear_name = next((name for name in (
        "ClearSkyGHIProxy", "ClearSkyGHI", "ClearSkyGTI",
        "SolarGeometryProxy", "ClearSkyProxy",
    ) if name in names), None)
    if clear_name is not None:
        ci = names.index(clear_name)
        mu = x.new_tensor(stats["feature_mu"])[ci]
        sd = x.new_tensor(stats["feature_sd"])[ci]
        clear = x[..., ci] * sd + mu
        clear_scale = max(float(stats.get("clear_sky_training_max") or 1.0), 1e-6)
        clear = clear / clear_scale
        daylight = (clear > 0.02).to(x.dtype)
        mismatch = ((target / cap - clear).abs() * daylight).sum(1) / daylight.sum(1).clamp_min(1.0)
        disturbance = disturbance + mismatch
    q20, q80 = torch.quantile(disturbance.detach(), x.new_tensor([0.2, 0.8]))
    state = ((disturbance - q20) / (q80 - q20 + 1e-6)).clamp(0.0, 1.0)
    quality_idx = stats.get("quality_indices") or []
    reliability = x[..., quality_idx].mean(dim=(1, 2)).clamp(0.0, 1.0) if quality_idx else torch.ones_like(state)

    # Scale order is fine -> coarse.  Build the target for the actual number
    # of physical scales instead of assuming a fixed three-scale backbone.
    scale_count = model.route_scale_weights.shape[-1]
    positions = torch.linspace(0.0, 1.0, scale_count, device=x.device, dtype=x.dtype)
    mid = 1.0 - (2.0 * state - 1.0).abs()
    end_preference = (
        (1.0 - positions[None, :]) * state[:, None]
        + positions[None, :] * (1.0 - state[:, None])
    )
    transition_preference = (
        4.0 * positions[None, :] * (1.0 - positions[None, :]) * mid[:, None]
    )
    scale_target = 0.1 + end_preference + transition_preference
    scale_target = scale_target / scale_target.sum(-1, keepdim=True)
    # Repeated sign reversals indicate oscillatory broken-cloud behaviour;
    # directed ramps are assigned to the temporal transition expert instead.
    if dx.shape[1] > 1:
        reversal = ((dx[:, 1:] * dx[:, :-1]) < 0).to(x.dtype).mean(1)
    else:
        reversal = torch.zeros_like(state)
    oscillatory = (2.0 * reversal).clamp(0.0, 1.0)
    temporal_score = 0.1 + state * (1.0 - 0.5 * oscillatory)
    frequency_score = (0.1 + state * oscillatory) * reliability
    if model.route_tf_weights.shape[-1] == 3:
        null_score = 0.1 + (1.0 - state)
        tf_target = torch.stack([temporal_score, frequency_score, null_score], dim=-1)
        tf_target = tf_target / tf_target.sum(-1, keepdim=True)
    else:
        tf_target = torch.stack([temporal_score, frequency_score], dim=-1)
        tf_target = tf_target / tf_target.sum(-1, keepdim=True)
    scale_weights = model.route_scale_weights.clamp_min(1e-6)
    scale_weights = scale_weights / scale_weights.sum(-1, keepdim=True)
    tf_weights = model.route_tf_weights.clamp_min(1e-6)
    tf_weights = tf_weights / tf_weights.sum(-1, keepdim=True)
    return (F.kl_div(scale_weights.log(), scale_target, reduction="batchmean")
            + F.kl_div(tf_weights.log(), tf_target, reduction="batchmean"))


def physical_state_auxiliary_loss(model, x, stats):
    """Separate operating regimes without prescribing router directions."""
    logits = getattr(model, "last_state_logits", None)
    if logits is None:
        return x.new_zeros(())
    cap = max(float(stats["capacity"]), 1e-6)
    target = x[..., -1] * stats["target_sd"] + stats["target_mu"]
    dx = torch.diff(target, dim=1)
    ramp = dx.abs().mean(1) / (cap * stats["dt_hours"] + 1e-6)
    variability = target.std(dim=1, unbiased=False) / cap
    directionality = dx.sum(1).abs() / dx.abs().sum(1).clamp_min(1e-6)
    q_ramp = torch.quantile(ramp.detach(), x.new_tensor([0.4, 0.6]))
    q_var = torch.quantile(variability.detach(), 0.5)

    # 0 stable, 1 directed ramp, 2 oscillatory/irregular, 3 low quality.
    label = torch.full_like(ramp, 2, dtype=torch.long)
    stable = (ramp <= q_ramp[0]) & (variability <= q_var)
    directed = (ramp >= q_ramp[1]) & (directionality >= 0.55)
    label[stable] = 0
    label[directed] = 1
    quality_idx = stats.get("quality_indices") or []
    if quality_idx:
        reliability = x[..., quality_idx].mean(dim=(1, 2))
        label[reliability < 0.8] = 3
    return F.cross_entropy(logits, label)


def routing_specialization_loss(model):
    """Encourage sample-wise specialization without assigning route meanings.

    Low per-sample entropy makes routing selective; a stronger batch-balance
    term prevents every sample from collapsing onto the same route.
    """
    weights = getattr(model, "route_tf_weights", None)
    if weights is None:
        return next(model.parameters()).new_zeros(())
    p = weights.clamp_min(1e-8)
    p = p / p.sum(dim=-1, keepdim=True)
    entropy = -(p * p.log()).sum(dim=-1).mean()
    mean_p = p.mean(dim=0)
    balance = (mean_p * (mean_p * p.shape[-1]).clamp_min(1e-8).log()).sum()
    return entropy + 2.0 * balance


def unpack_batch(batch, device):
    if len(batch) == 6:
        x, y, y_mask, cycle, future_x, prompt_x = batch
        future_x = None if future_x.shape[-1] == 0 else future_x.to(device)
        return (x.to(device), y.to(device), y_mask.to(device), cycle.to(device),
                future_x, prompt_x.to(device))
    if len(batch) == 5:
        x, y, y_mask, cycle, last = batch
        # In the benchmark every prompt model also uses the audited solar
        # anchor and therefore has six fields.  Five fields unambiguously mean
        # a future-covariate batch.
        return x.to(device), y.to(device), y_mask.to(device), cycle.to(device), last.to(device), None
    x, y, y_mask, cycle = batch
    return x.to(device), y.to(device), y_mask.to(device), cycle.to(device), None, None


def predict(model, x, cycle, future_x=None, prompt_x=None):
    # 源码风格模型统一使用标准预测接口，并在实验层压缩单目标维度。
    if getattr(model, "source_forecast_interface", False):
        output = model(x, cycle, future_x, prompt_x)
        if output.ndim == 3 and output.shape[-1] == 1:
            output = output.squeeze(-1)
        return output

    # A prepared prompt batch is itself the authoritative interface signal.
    # Some staged prompt models keep their reported semantic dimension at zero
    # until the adapter is enabled, even though their forward method already
    # requires the prompt argument for paired baseline evaluation.
    if prompt_x is not None:
        return model(x, cycle, future_x, prompt_x)
    if future_x is None or bool(
        getattr(model, "projection_only_future", False)
    ):
        return model(x, cycle)
    return model(x, cycle, future_x)


def _masked_metric_block(y, p, y_phys, p_phys, selected, cap):
    """Metrics for one target-independent reporting mask."""
    standardized_error = p - y
    physical_error = p_phys - y_phys
    selected_y = y_phys[selected]
    denom = float(np.sum((selected_y - selected_y.mean()) ** 2))
    shortfall = np.maximum(-physical_error[selected], 0.0)
    surplus = np.maximum(physical_error[selected], 0.0)
    return {
        "mse_standardized": float(np.mean(standardized_error[selected] ** 2)),
        "mae_standardized": float(np.mean(np.abs(standardized_error[selected]))),
        "rmse_physical": float(np.sqrt(np.mean(physical_error[selected] ** 2))),
        "mae_physical": float(np.mean(np.abs(physical_error[selected]))),
        "mbe_physical": float(np.mean(physical_error[selected])),
        "r2": 1.0 - float(np.sum(physical_error[selected] ** 2)) / max(denom, 1e-12),
        # Dimensionless imbalance-energy proxies.  A value of 0.01 means the
        # mean per-step imbalance cost equals 1% of rated-capacity energy for
        # one sampling interval.  The asymmetric cases expose conclusions that
        # depend on shortage/surplus settlement assumptions without inventing
        # a site-specific currency price.
        "imbalance_cost_equal_norm": float(np.mean(shortfall + surplus)) / cap,
        "imbalance_cost_short2_norm": float(np.mean(2.0 * shortfall + surplus)) / cap,
        "imbalance_cost_surplus2_norm": float(np.mean(shortfall + 2.0 * surplus)) / cap,
    }


def metrics(y, p, stats, mask=None, operational_mask=None,
            solar_daylight_mask=None):
    y = np.asarray(y); p = np.asarray(p)
    mask = np.ones_like(y, dtype=bool) if mask is None else np.asarray(mask) > 0.5
    if not mask.any():
        raise ValueError("No observed target values are available for evaluation")
    y_phys = y * stats["target_sd"] + stats["target_mu"]
    p_phys = p * stats["target_sd"] + stats["target_mu"]
    err = p_phys - y_phys
    standardized_error = p - y
    mse = float(np.mean((standardized_error ** 2)[mask]))
    mae_standardized = float(np.mean(np.abs(standardized_error)[mask]))
    mbe_standardized = float(np.mean(standardized_error[mask]))
    rmse = float(np.sqrt(np.mean((err ** 2)[mask])))
    mae = float(np.mean(np.abs(err)[mask]))
    mbe = float(np.mean(err[mask]))
    cap = max(float(stats["capacity"]), 1e-6)
    y_obs = y_phys[mask]
    denom = float(np.sum((y_obs - y_obs.mean()) ** 2))
    r2 = 1.0 - float(np.sum((err[mask]) ** 2)) / max(denom, 1e-12)
    if y.shape[1] > 1:
        ramp = np.diff(p_phys, axis=1) - np.diff(y_phys, axis=1)
        ramp_mask = mask[:, 1:] & mask[:, :-1]
        ramp_mae = (float(np.mean(np.abs(ramp)[ramp_mask])) /
                    (cap * stats["dt_hours"] + 1e-6)) if ramp_mask.any() else 0.0
    else:
        ramp_mae = 0.0
    # Report the conventional all-period score and a separate daylight score.
    # PV datasets contain long deterministic night plateaus; mixing those
    # points into the headline MSE can make a strong daily-shape forecaster
    # appear to have an implausibly small error compared with daytime-only PV
    # studies or multivariate LTSF benchmarks.  The daylight mask never enters
    # model inputs.  It may be used on the validation split for explicitly
    # configured checkpoint selection, and on training labels for an optional
    # mean-one weighted point loss; test targets remain reporting-only.
    daylight = (y_phys > 0.01 * cap) & mask
    if daylight.any():
        daylight_standardized_error = standardized_error[daylight]
        daylight_physical_error = err[daylight]
        daylight_mse_standardized = float(np.mean(daylight_standardized_error ** 2))
        daylight_mae_standardized = float(np.mean(np.abs(daylight_standardized_error)))
        daylight_rmse = float(np.sqrt(np.mean(daylight_physical_error ** 2)))
        daylight_mae = float(np.mean(np.abs(daylight_physical_error)))
        daylight_mbe = float(np.mean(daylight_physical_error))
    else:
        daylight_mse_standardized = 0.0
        daylight_mae_standardized = 0.0
        daylight_rmse = 0.0
        daylight_mae = 0.0
        daylight_mbe = 0.0
    result = {"mse_standardized": mse, "mae_standardized": mae_standardized,
            "mbe_standardized": mbe_standardized,
            "rmse_physical": rmse, "mae_physical": mae,
            "mbe_physical": mbe,
            "nrmse_capacity": rmse / cap, "nmae_capacity": mae / cap,
            "nmbe_capacity": mbe / cap,
            "r2": r2, "ramp_mae_norm": ramp_mae,
            "daylight_mse_standardized": daylight_mse_standardized,
            "daylight_mae_standardized": daylight_mae_standardized,
            "daylight_rmse_physical": daylight_rmse,
            "daylight_mae_physical": daylight_mae,
            "daylight_mbe_physical": daylight_mbe,
            "daylight_nrmse_capacity": daylight_rmse / cap,
            "daylight_nmae_capacity": daylight_mae / cap,
            "daylight_fraction": float(daylight.sum() / mask.sum()),
            "observed_target_fraction": float(mask.mean()),
            "observed_target_count": int(mask.sum())}
    all_block = _masked_metric_block(y, p, y_phys, p_phys, mask, cap)
    for key in (
        "imbalance_cost_equal_norm", "imbalance_cost_short2_norm",
        "imbalance_cost_surplus2_norm",
    ):
        result[key] = all_block[key] * float(stats["dt_hours"])
    if operational_mask is not None:
        operational = (np.asarray(operational_mask) > 0.5) & mask
        if operational.any():
            operational_error = standardized_error[operational]
            operational_physical_error = err[operational]
            operational_y = y_phys[operational]
            operational_denom = float(np.sum(
                (operational_y - operational_y.mean()) ** 2
            ))
            result.update({
                "operational_0619_mse_standardized": float(np.mean(
                    operational_error ** 2
                )),
                "operational_0619_mae_standardized": float(np.mean(
                    np.abs(operational_error)
                )),
                "operational_0619_rmse_physical": float(np.sqrt(np.mean(
                    operational_physical_error ** 2
                ))),
                "operational_0619_mae_physical": float(np.mean(
                    np.abs(operational_physical_error)
                )),
                "operational_0619_mbe_physical": float(np.mean(
                    operational_physical_error
                )),
                "operational_0619_r2": 1.0 - float(np.sum(
                    operational_physical_error ** 2
                )) / max(operational_denom, 1e-12),
                "operational_0619_fraction": float(
                    operational.sum() / mask.sum()
                ),
            })
            result["operational_0619_nrmse_capacity"] = (
                result["operational_0619_rmse_physical"] / cap
            )
            result["operational_0619_nmae_capacity"] = (
                result["operational_0619_mae_physical"] / cap
            )
    if solar_daylight_mask is not None:
        solar_daylight = (np.asarray(solar_daylight_mask) > 0.5) & mask
        if solar_daylight.any():
            block = _masked_metric_block(
                y, p, y_phys, p_phys, solar_daylight, cap,
            )
            for key, value in block.items():
                result[f"solar_daylight_{key}"] = value
            result["solar_daylight_nrmse_capacity"] = (
                block["rmse_physical"] / cap
            )
            result["solar_daylight_nmae_capacity"] = (
                block["mae_physical"] / cap
            )
            result["solar_daylight_fraction"] = float(
                solar_daylight.sum() / mask.sum()
            )
            for key in (
                "imbalance_cost_equal_norm", "imbalance_cost_short2_norm",
                "imbalance_cost_surplus2_norm",
            ):
                result[f"solar_daylight_{key}"] = (
                    block[key] * float(stats["dt_hours"])
                )
    return result


def _regime_breakdown(xs, ys, ps, masks, stats, scale_ws, tf_ws, correction_ws,
                      spectral_ws):
    """Test the claimed physics-routing mechanism on sample-level regimes."""
    names = stats["feature_names"]
    mu = np.asarray(stats["feature_mu"], dtype=np.float32)
    sd = np.asarray(stats["feature_sd"], dtype=np.float32)
    cap = max(float(stats["capacity"]), 1e-6)
    target_hist = xs[..., -1] * stats["target_sd"] + stats["target_mu"]
    dx = np.diff(target_hist, axis=1)
    ramp = np.mean(np.abs(dx), axis=1) / (cap * stats["dt_hours"] + 1e-6)
    directionality = np.abs(dx.sum(axis=1)) / np.maximum(np.abs(dx).sum(axis=1), 1e-6)
    oscillation = np.mean((dx[:, 1:] * dx[:, :-1]) < 0, axis=1) if dx.shape[1] > 1 else np.zeros_like(ramp)
    indicators = {
        "historical_ramp_intensity": ramp,
        "change_directionality": directionality,
        "oscillation_index": oscillation,
    }
    qidx = stats.get("quality_indices") or []
    if qidx:
        indicators["observed_fraction"] = xs[..., qidx].mean(axis=(1, 2))
    clear_name = next((name for name in ("ClearSkyGHIProxy", "SolarGeometryProxy", "ClearSkyProxy")
                       if name in names), None)
    if clear_name is not None:
        ci = names.index(clear_name)
        clear = xs[..., ci] * sd[ci] + mu[ci]
        if clear_name == "ClearSkyGHIProxy":
            clear_norm = clear / max(float(np.nanmax(clear)), 1e-6)
            daylight = clear > 20.0
        else:
            clear_norm = clear
            daylight = clear > 0.02
        mismatch = np.abs(target_hist / cap - clear_norm)
        indicators["clear_sky_deviation"] = np.sum(mismatch * daylight, axis=1) / np.maximum(daylight.sum(axis=1), 1)

    masks = np.asarray(masks, dtype=np.float32)
    denom = np.maximum(masks.sum(axis=1), 1.0)
    sample_mse = (((ps - ys) ** 2) * masks).sum(axis=1) / denom
    sample_nmae = ((np.abs((ps - ys) * stats["target_sd"]) * masks).sum(axis=1) /
                   denom / cap)
    scales = np.concatenate(scale_ws) if scale_ws else None
    tf = np.concatenate(tf_ws) if tf_ws else None
    corr = np.concatenate(correction_ws) if correction_ws else None
    spectral = np.concatenate(spectral_ws) if spectral_ws else None
    result = {}
    for name, values in indicators.items():
        lo, hi = np.quantile(values, [1 / 3, 2 / 3])
        masks = {"low": values <= lo, "mid": (values > lo) & (values <= hi), "high": values > hi}
        groups = {}
        for label, mask in masks.items():
            if not mask.any():
                continue
            row = {"count": int(mask.sum()), "indicator_mean": float(values[mask].mean()),
                   "mse_standardized": float(sample_mse[mask].mean()),
                   "nmae_capacity": float(sample_nmae[mask].mean())}
            if scales is not None:
                row["scale_weights"] = scales[mask].mean(axis=0).tolist()
            if tf is not None:
                row["time_frequency_weights"] = tf[mask].mean(axis=0).tolist()
            if corr is not None:
                row["output_correction_mean"] = float(corr[mask].mean())
            if spectral is not None and spectral.shape[1] == 3:
                row["spectral_band_weights"] = spectral[mask].mean(axis=0).tolist()
            groups[label] = row
        result[name] = {"tertile_edges": [float(lo), float(hi)], "groups": groups}
    return result


def future_solar_daylight_mask(future_x, stats):
    """Return a target-free daylight mask from deterministic solar geometry."""
    if future_x is None:
        return None
    future_names = stats.get("solar_future_features") or []
    daylight_name = stats.get("daylight_feature")
    clear_name = stats.get("clear_sky_feature")
    if daylight_name in future_names:
        source_idx = stats["feature_names"].index(daylight_name)
        source = (
            future_x[..., future_names.index(daylight_name)]
            * stats["feature_sd"][source_idx]
            + stats["feature_mu"][source_idx]
        )
        return source > 0.5
    if clear_name in future_names:
        source_idx = stats["feature_names"].index(clear_name)
        source = (
            future_x[..., future_names.index(clear_name)]
            * stats["feature_sd"][source_idx]
            + stats["feature_mu"][source_idx]
        )
        threshold = 0.02 * max(
            float(stats.get("clear_sky_training_max") or 1.0), 1e-6,
        )
        return source > threshold
    return None


def project_pv_prediction(pred, future_x, stats, mode="none"):
    """Project standardized PV predictions onto a causal physical feasible set.

    The projection consumes no target value.  ``nonnegative_night`` uses only
    the zero-power lower bound and deterministic future solar geometry already
    audited for the physical reference.  It deliberately does not clip to the
    training empirical capacity: a future observation may legitimately exceed
    a training-set maximum.
    """
    if mode == "none":
        return pred
    if mode not in {"nonnegative", "nonnegative_night"}:
        raise ValueError(f"unknown evaluation projection: {mode}")
    zero = pred.new_tensor(
        (0.0 - float(stats["target_mu"])) / float(stats["target_sd"])
    )
    projected = pred.clamp_min(zero)
    if mode == "nonnegative":
        return projected
    if future_x is None:
        raise ValueError(
            "night projection requires audited deterministic future solar geometry"
        )
    daylight = future_solar_daylight_mask(future_x, stats)
    if daylight is None:
        raise ValueError(
            "night projection requires a daylight or clear-sky future feature"
        )
    night = ~daylight
    return torch.where(night, zero, projected)


@torch.no_grad()
def evaluate(model, loader, device, stats, analyze_regimes=False,
             include_preview=False):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    evaluation_started = time.perf_counter()
    model.eval(); ys = []; ps = []; raw_ps = []; masks = []; operational_masks = []; solar_daylight_masks = []; candidate_ws = []; xs = []; scale_ws = []; tf_ws = []; correction_ws = []; spectral_ws = []; frequency_v2_gates = []; frequency_v2_scales = []; frequency_v2_ratios = []; frequency_v2_energies = []; frequency_v3_gates = []; frequency_v3_scales = []; frequency_v3_ratios = []; frequency_v3_energies = []; frequency_v4_gates = []; frequency_v4_scales = []; frequency_v4_ratios = []; frequency_v4_energies = []; frequency_v4_local_gates = []; frequency_v5_nulls = []; frequency_v5_channel_gates = []; frequency_v5_scales = []; frequency_v5_ratios = []; frequency_v6_gates = []; frequency_v6_energies = []; frequency_v6_scales = []; frequency_v6_ratios = []; frequency_v6_magnitude = []; frequency_v6_phase = []; frequency_v7_gates = []; frequency_v7_coherences = []; frequency_v7_stabilities = []; frequency_v7_priors = []; frequency_v7_clips = []; frequency_v7_ratios = []; frequency_v7_magnitude = []; frequency_v7_phase = []; frequency_v8_gates = []; frequency_v8_energies = []; frequency_v8_robust_clips = []; frequency_v8_caps = []; frequency_v8_residual_clips = []; frequency_v8_raw_ratios = []; frequency_v8_ratios = []; frequency_v8_channel_ratios = []; frequency_v9_trend_mix = []; frequency_v9_gates = []; frequency_v9_scales = []; frequency_v9_trend_energy = []; frequency_v9_raw_ratios = []; frequency_v9_ratios = []; frequency_v9_target_ratios = []; frequency_v9_robust_clips = []; semantic_ws = []; semantic_head_ws = []; scale_prompt_ws = []; scale_prompt_head_ws = []; scale_prompt_match = []; fsra_ws = []; fsra_head_ws = []; predictive_scale_ws = []; predictive_member_ws = []; night_ws = []
    spectral_residual_gains = []
    spectral_raw_ratios = []
    spectral_ratios = []
    for batch in loader:
        x, y, y_mask, cycle, future_x, prompt_x = unpack_batch(batch, device)
        raw_pred = predict(model, x, cycle, future_x, prompt_x)
        pred = project_pv_prediction(
            raw_pred, future_x, stats,
            mode=getattr(model, "evaluation_projection", "none"),
        )
        if getattr(model, "last_candidate_pred", None) is not None:
            candidate_ws.append(model.last_candidate_pred.cpu().numpy())
        if getattr(model, "last_scale_weights", None) is not None:
            scale_ws.append(model.last_scale_weights.cpu().numpy())
        if getattr(model, "last_tf_weights", None) is not None:
            tf_ws.append(model.last_tf_weights.cpu().numpy())
        if getattr(model, "last_output_correction", None) is not None:
            correction_ws.append(model.last_output_correction.cpu().numpy())
        if getattr(model, "last_spectral_attention", None) is not None:
            spectral_ws.append(model.last_spectral_attention.cpu().numpy())
        if getattr(model, "last_spectral_residual_gain", None) is not None:
            spectral_residual_gains.append(
                float(model.last_spectral_residual_gain.cpu())
            )
        if getattr(model, "last_spectral_raw_contribution_ratio", None) is not None:
            spectral_raw_ratios.append(
                model.last_spectral_raw_contribution_ratio.cpu().numpy()
            )
        if getattr(model, "last_spectral_contribution_ratio", None) is not None:
            spectral_ratios.append(
                model.last_spectral_contribution_ratio.cpu().numpy()
            )
        if getattr(model, "last_frequency_v2_gate", None) is not None:
            frequency_v2_gates.append(
                model.last_frequency_v2_gate.cpu().numpy()
            )
        if getattr(model, "last_frequency_v2_scale", None) is not None:
            frequency_v2_scales.append(
                model.last_frequency_v2_scale.cpu().numpy()
            )
        if getattr(model, "last_frequency_v2_contribution_ratio", None) is not None:
            frequency_v2_ratios.append(
                model.last_frequency_v2_contribution_ratio.cpu().numpy()
            )
        if getattr(model, "last_frequency_v2_source_energy", None) is not None:
            frequency_v2_energies.append(
                model.last_frequency_v2_source_energy.cpu().numpy()
            )
        if getattr(model, "last_frequency_v3_gate", None) is not None:
            frequency_v3_gates.append(
                model.last_frequency_v3_gate.cpu().numpy()
            )
        if getattr(model, "last_frequency_v3_scale", None) is not None:
            frequency_v3_scales.append(
                float(model.last_frequency_v3_scale.cpu())
            )
        if getattr(model, "last_frequency_v3_contribution_ratio", None) is not None:
            frequency_v3_ratios.append(
                model.last_frequency_v3_contribution_ratio.cpu().numpy()
            )
        if getattr(model, "last_frequency_v3_spectral_energy", None) is not None:
            frequency_v3_energies.append(
                model.last_frequency_v3_spectral_energy.cpu().numpy()
            )
        if getattr(model, "last_frequency_v4_gate", None) is not None:
            frequency_v4_gates.append(model.last_frequency_v4_gate.cpu().numpy())
        if getattr(model, "last_frequency_v4_scale", None) is not None:
            frequency_v4_scales.append(model.last_frequency_v4_scale.cpu().numpy())
        if getattr(model, "last_frequency_v4_contribution_ratio", None) is not None:
            frequency_v4_ratios.append(
                model.last_frequency_v4_contribution_ratio.cpu().numpy()
            )
        if getattr(model, "last_frequency_v4_source_energy", None) is not None:
            frequency_v4_energies.append(
                model.last_frequency_v4_source_energy.cpu().numpy()
            )
        if getattr(model, "last_frequency_v4_local_gate", None) is not None:
            frequency_v4_local_gates.append(
                model.last_frequency_v4_local_gate.cpu().numpy()
            )
        if getattr(model, "last_frequency_v5_null_probability", None) is not None:
            frequency_v5_nulls.append(
                model.last_frequency_v5_null_probability.cpu().numpy()
            )
        if getattr(model, "last_frequency_v5_channel_gate", None) is not None:
            frequency_v5_channel_gates.append(
                model.last_frequency_v5_channel_gate.cpu().numpy()
            )
        if getattr(model, "last_frequency_v5_residual_scale", None) is not None:
            frequency_v5_scales.append(
                float(model.last_frequency_v5_residual_scale.cpu())
            )
        if getattr(model, "last_frequency_v5_contribution_ratio", None) is not None:
            frequency_v5_ratios.append(
                model.last_frequency_v5_contribution_ratio.cpu().numpy()
            )
        if getattr(model, "last_frequency_v6_band_gate", None) is not None:
            frequency_v6_gates.append(
                model.last_frequency_v6_band_gate.cpu().numpy()
            )
        if getattr(model, "last_frequency_v6_band_energy_fraction", None) is not None:
            frequency_v6_energies.append(
                model.last_frequency_v6_band_energy_fraction.cpu().numpy()
            )
        if getattr(model, "last_frequency_v6_residual_scale", None) is not None:
            frequency_v6_scales.append(
                float(model.last_frequency_v6_residual_scale.cpu())
            )
        if getattr(model, "last_frequency_v6_contribution_ratio", None) is not None:
            frequency_v6_ratios.append(
                model.last_frequency_v6_contribution_ratio.cpu().numpy()
            )
        if getattr(model, "last_frequency_v6_magnitude_adjustment", None) is not None:
            frequency_v6_magnitude.append(
                model.last_frequency_v6_magnitude_adjustment.cpu().numpy()
            )
        if getattr(model, "last_frequency_v6_phase_adjustment", None) is not None:
            frequency_v6_phase.append(
                model.last_frequency_v6_phase_adjustment.cpu().numpy()
            )
        if getattr(model, "last_frequency_v7_band_gate", None) is not None:
            frequency_v7_gates.append(model.last_frequency_v7_band_gate.cpu().numpy())
        if getattr(model, "last_frequency_v7_coherence", None) is not None:
            frequency_v7_coherences.append(model.last_frequency_v7_coherence.cpu().numpy())
        if getattr(model, "last_frequency_v7_stability", None) is not None:
            frequency_v7_stabilities.append(model.last_frequency_v7_stability.cpu().numpy())
        if getattr(model, "last_frequency_v7_horizon_prior", None) is not None:
            frequency_v7_priors.append(model.last_frequency_v7_horizon_prior.cpu().numpy())
        if getattr(model, "last_frequency_v7_clip_factor", None) is not None:
            frequency_v7_clips.append(model.last_frequency_v7_clip_factor.cpu().numpy())
        if getattr(model, "last_frequency_v7_contribution_ratio", None) is not None:
            frequency_v7_ratios.append(model.last_frequency_v7_contribution_ratio.cpu().numpy())
        if getattr(model, "last_frequency_v7_magnitude_adjustment", None) is not None:
            frequency_v7_magnitude.append(model.last_frequency_v7_magnitude_adjustment.cpu().numpy())
        if getattr(model, "last_frequency_v7_phase_adjustment", None) is not None:
            frequency_v7_phase.append(model.last_frequency_v7_phase_adjustment.cpu().numpy())
        if getattr(model, "last_frequency_v8_band_channel_gate", None) is not None:
            frequency_v8_gates.append(model.last_frequency_v8_band_channel_gate.cpu().numpy())
        if getattr(model, "last_frequency_v8_source_energy_fraction", None) is not None:
            frequency_v8_energies.append(model.last_frequency_v8_source_energy_fraction.cpu().numpy())
        if getattr(model, "last_frequency_v8_robust_clip_fraction", None) is not None:
            frequency_v8_robust_clips.append(model.last_frequency_v8_robust_clip_fraction.cpu().numpy())
        if getattr(model, "last_frequency_v8_residual_cap", None) is not None:
            frequency_v8_caps.append(model.last_frequency_v8_residual_cap.cpu().numpy())
        if getattr(model, "last_frequency_v8_residual_clip_fraction", None) is not None:
            frequency_v8_residual_clips.append(model.last_frequency_v8_residual_clip_fraction.cpu().numpy())
        if getattr(model, "last_frequency_v8_raw_contribution_ratio", None) is not None:
            frequency_v8_raw_ratios.append(model.last_frequency_v8_raw_contribution_ratio.cpu().numpy())
        if getattr(model, "last_frequency_v8_contribution_ratio", None) is not None:
            frequency_v8_ratios.append(model.last_frequency_v8_contribution_ratio.cpu().numpy())
        if getattr(model, "last_frequency_v8_channel_contribution_ratio", None) is not None:
            frequency_v8_channel_ratios.append(model.last_frequency_v8_channel_contribution_ratio.cpu().numpy())
        if getattr(model, "last_frequency_v9_trend_mix", None) is not None:
            frequency_v9_trend_mix.append(model.last_frequency_v9_trend_mix.cpu().numpy())
        if getattr(model, "last_frequency_v9_band_gate", None) is not None:
            frequency_v9_gates.append(model.last_frequency_v9_band_gate.cpu().numpy())
        if getattr(model, "last_frequency_v9_residual_scale", None) is not None:
            frequency_v9_scales.append(model.last_frequency_v9_residual_scale.cpu().numpy())
        if getattr(model, "last_frequency_v9_trend_energy_fraction", None) is not None:
            frequency_v9_trend_energy.append(model.last_frequency_v9_trend_energy_fraction.cpu().numpy())
        if getattr(model, "last_frequency_v9_raw_contribution_ratio", None) is not None:
            frequency_v9_raw_ratios.append(model.last_frequency_v9_raw_contribution_ratio.cpu().numpy())
        if getattr(model, "last_frequency_v9_contribution_ratio", None) is not None:
            frequency_v9_ratios.append(model.last_frequency_v9_contribution_ratio.cpu().numpy())
        if getattr(model, "last_frequency_v9_target_contribution_ratio", None) is not None:
            frequency_v9_target_ratios.append(model.last_frequency_v9_target_contribution_ratio.cpu().numpy())
        if getattr(model, "last_frequency_v9_robust_clip_fraction", None) is not None:
            frequency_v9_robust_clips.append(model.last_frequency_v9_robust_clip_fraction.cpu().numpy())
        if getattr(model, "last_semantic_gate", None) is not None:
            semantic_ws.append(model.last_semantic_gate.cpu().numpy())
        if getattr(model, "last_semantic_head_weights", None) is not None:
            semantic_head_ws.append(model.last_semantic_head_weights.cpu().numpy())
        if getattr(model, "last_scale_prompt_gate", None) is not None:
            scale_prompt_ws.append(model.last_scale_prompt_gate.cpu().numpy())
        if getattr(model, "last_scale_prompt_head_weights", None) is not None:
            scale_prompt_head_ws.append(
                model.last_scale_prompt_head_weights.cpu().numpy()
            )
        if getattr(model, "last_scale_prompt_match_accuracy", None) is not None:
            scale_prompt_match.append(
                float(model.last_scale_prompt_match_accuracy.cpu())
            )
        if getattr(model, "last_fsra_gate", None) is not None:
            fsra_ws.append(model.last_fsra_gate.cpu().numpy())
        if getattr(model, "last_fsra_head_weights", None) is not None:
            fsra_head_ws.append(model.last_fsra_head_weights.cpu().numpy())
        if getattr(model, "last_predictive_scale", None) is not None:
            predictive_scale_ws.append(model.last_predictive_scale.cpu().numpy())
            night = torch.zeros_like(y, dtype=torch.bool)
            if future_x is not None:
                future_names = stats.get("solar_future_features") or []
                daylight_name = stats.get("daylight_feature")
                clear_name = stats.get("clear_sky_feature")
                if daylight_name in future_names:
                    source_idx = stats["feature_names"].index(daylight_name)
                    source = (
                        future_x[..., future_names.index(daylight_name)]
                        * stats["feature_sd"][source_idx]
                        + stats["feature_mu"][source_idx]
                    )
                    night = source <= 0.5
                elif clear_name in future_names:
                    source_idx = stats["feature_names"].index(clear_name)
                    source = (
                        future_x[..., future_names.index(clear_name)]
                        * stats["feature_sd"][source_idx]
                        + stats["feature_mu"][source_idx]
                    )
                    threshold = 0.02 * max(
                        float(stats.get("clear_sky_training_max") or 1.0), 1e-6,
                    )
                    night = source <= threshold
            night_ws.append(night.cpu().numpy())
        if getattr(model, "last_predictive_members", None) is not None:
            predictive_member_ws.append(
                model.last_predictive_members.cpu().numpy()
            )
        if pred.shape[1] == 1 and y.shape[1] > 1:
            pred = pred.expand(-1, y.shape[1])
        if raw_pred.shape[1] == 1 and y.shape[1] > 1:
            raw_pred = raw_pred.expand(-1, y.shape[1])
        # Target-free 06:00--19:00 reporting mask used by the 2026 AMPDNet
        # study.  Derive it from the audited window-start clock, so it never
        # becomes an additional forecasting input channel.
        phase = (
            cycle[:, None]
            + int(stats["seq_len"])
            + torch.arange(y.shape[1], device=cycle.device)[None, :]
        ) % int(stats["cycle_len"])
        lower = int(round(6.0 / 24.0 * int(stats["cycle_len"])))
        upper = int(round(19.0 / 24.0 * int(stats["cycle_len"])))
        operational_masks.append(((phase >= lower) & (phase <= upper)).cpu().numpy())
        solar_daylight = future_solar_daylight_mask(future_x, stats)
        if solar_daylight is not None:
            solar_daylight_masks.append(solar_daylight.cpu().numpy())
        ys.append(y.cpu().numpy()); ps.append(pred.cpu().numpy()); raw_ps.append(raw_pred.cpu().numpy()); masks.append(y_mask.cpu().numpy())
        if analyze_regimes and hasattr(model, "last_scale_weights"):
            xs.append(x.cpu().numpy())
    y_all, p_all, raw_p_all, mask_all = (
        np.concatenate(ys), np.concatenate(ps), np.concatenate(raw_ps),
        np.concatenate(masks),
    )
    operational_all = (
        np.concatenate(operational_masks) if operational_masks else None
    )
    solar_daylight_all = (
        np.concatenate(solar_daylight_masks) if solar_daylight_masks else None
    )
    result = metrics(
        y_all, p_all, stats, mask_all, operational_all, solar_daylight_all,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    evaluation_seconds = max(time.perf_counter() - evaluation_started, 1e-12)
    result.update({
        "evaluation_seconds": float(evaluation_seconds),
        "evaluation_windows": int(y_all.shape[0]),
        "evaluation_values": int(y_all.size),
        "evaluation_windows_per_second": float(y_all.shape[0] / evaluation_seconds),
        "evaluation_values_per_second": float(y_all.size / evaluation_seconds),
    })
    raw_result = metrics(
        y_all, raw_p_all, stats, mask_all, operational_all,
        solar_daylight_all,
    )
    result["raw_unprojected"] = raw_result
    if candidate_ws:
        candidate_all = np.concatenate(candidate_ws)
        candidate_result = metrics(
            y_all, candidate_all, stats, mask_all, operational_all,
        )
        result["candidate_mse_standardized"] = candidate_result["mse_standardized"]
        result["candidate_mae_standardized"] = candidate_result["mae_standardized"]
    if include_preview:
        # Select an information-rich, fully observed example without using its
        # error. The same target-variance rule is applied to every model, so
        # curve selection cannot favor a particular prediction.
        observed_fraction = mask_all.mean(axis=1)
        target_variance = np.var(y_all, axis=1)
        score = np.where(observed_fraction >= 0.999, target_variance, -np.inf)
        preview_index = int(np.argmax(score))
        y_preview = y_all[preview_index]
        p_preview = p_all[preview_index]
        result["preview"] = {
            "sample_index": preview_index,
            "target_standardized": y_preview.astype(float).tolist(),
            "prediction_standardized": p_preview.astype(float).tolist(),
            "target_physical": (
                y_preview * stats["target_sd"] + stats["target_mu"]
            ).astype(float).tolist(),
            "prediction_physical": (
                p_preview * stats["target_sd"] + stats["target_mu"]
            ).astype(float).tolist(),
        }
    if predictive_scale_ws:
        predictive_scale = np.concatenate(predictive_scale_ws)
        scale = np.maximum(predictive_scale, 1e-6)
        z = (y_all - p_all) / scale
        z_tensor = torch.from_numpy(z)
        phi = torch.exp(-0.5 * z_tensor.square()) / np.sqrt(2.0 * np.pi)
        cdf = 0.5 * (1.0 + torch.erf(z_tensor / np.sqrt(2.0)))
        crps = scale * (
            z * (2.0 * cdf.numpy() - 1.0)
            + 2.0 * phi.numpy() - 1.0 / np.sqrt(np.pi)
        )
        observed = mask_all > 0.5
        z90 = 1.6448536269514722
        lower = p_all - z90 * scale
        upper = p_all + z90 * scale
        nll = 0.5 * z ** 2 + np.log(scale) + 0.5 * np.log(2.0 * np.pi)
        crps_std = float(np.mean(crps[observed]))
        result.update({
            "gaussian_crps_standardized": crps_std,
            "gaussian_crps_physical": crps_std * float(stats["target_sd"]),
            "gaussian_nll_standardized": float(np.mean(nll[observed])),
            "prediction_interval_90_coverage": float(np.mean(
                ((y_all >= lower) & (y_all <= upper))[observed]
            )),
            "prediction_interval_90_width_physical": float(np.mean(
                (upper - lower)[observed]
            )) * float(stats["target_sd"]),
            "predictive_scale_mean_standardized": float(np.mean(scale[observed])),
            "crps_skill_vs_point_forecast_pct": 100.0 * (
                result["mae_standardized"] - crps_std
            ) / max(result["mae_standardized"], 1e-8),
        })
        # Diagnose whether state conditioning changes uncertainty across
        # samples rather than learning only a horizon-wise climatology.
        observed_scale = scale.copy()
        observed_error = np.abs(y_all - p_all)
        for horizon_index in range(scale.shape[1]):
            horizon_mask = observed[:, horizon_index]
            if horizon_mask.any():
                observed_scale[horizon_mask, horizon_index] -= observed_scale[
                    horizon_mask, horizon_index
                ].mean()
                observed_error[horizon_mask, horizon_index] -= observed_error[
                    horizon_mask, horizon_index
                ].mean()
        scale_flat = observed_scale[observed]
        error_flat = observed_error[observed]
        scale_std = float(np.std(scale_flat))
        error_std = float(np.std(error_flat))
        scale_error_corr = (
            float(np.mean((scale_flat - scale_flat.mean()) * (error_flat - error_flat.mean()))
                  / max(scale_std * error_std, 1e-12))
            if scale_std > 1e-8 and error_std > 1e-8 else 0.0
        )
        result["predictive_scale_sample_std_standardized"] = scale_std
        result["predictive_scale_abs_error_correlation"] = scale_error_corr

        # A deterministic 19-member normal-quantile ensemble makes the output
        # auditable and reproducible. Members are projected to the PV feasible
        # capacity set. Night behavior remains governed by the differentiable
        # solar-mask loss: hard zeroing would destroy empirical coverage when
        # meters retain small night-time offsets.
        probabilities = torch.arange(1, 20, dtype=torch.float64) / 20.0
        normal_quantiles = torch.distributions.Normal(0.0, 1.0).icdf(
            probabilities
        ).numpy().astype(np.float32)
        ensemble = p_all[None, ...] + normal_quantiles[:, None, None] * scale[None, ...]
        ensemble_phys = ensemble * float(stats["target_sd"]) + float(stats["target_mu"])
        ensemble_phys = np.clip(ensemble_phys, 0.0, float(stats["capacity"]))
        ensemble_std = (
            ensemble_phys - float(stats["target_mu"])
        ) / max(float(stats["target_sd"]), 1e-8)
        first_term = np.mean(np.abs(ensemble_std - y_all[None, ...]), axis=0)
        coefficients = (2.0 * np.arange(1, 20) - 20.0).astype(np.float32)
        second_term = np.sum(
            coefficients[:, None, None] * ensemble_std, axis=0,
        ) / (19.0 ** 2)
        physical_crps = first_term - second_term
        physical_crps_std = float(np.mean(physical_crps[observed]))
        lower_phys = ensemble_phys[0]
        upper_phys = ensemble_phys[-1]
        y_phys = y_all * float(stats["target_sd"]) + float(stats["target_mu"])
        result.update({
            "physical_ensemble_members": 19,
            "physical_ensemble_crps_standardized": physical_crps_std,
            "physical_ensemble_crps_physical": physical_crps_std * float(stats["target_sd"]),
            "physical_ensemble_crps_skill_vs_point_pct": 100.0 * (
                result["mae_standardized"] - physical_crps_std
            ) / max(result["mae_standardized"], 1e-8),
            "physical_ensemble_interval_90_coverage": float(np.mean(
                ((y_phys >= lower_phys) & (y_phys <= upper_phys))[observed]
            )),
            "physical_ensemble_interval_90_width_physical": float(np.mean(
                (upper_phys - lower_phys)[observed]
            )),
        })
    if predictive_member_ws:
        # [N,H,M] -> sorted [M,N,H].  Quantile levels are 0.05,...,0.95
        # for M=19, so the first/last members form the nominal 90% interval.
        member_nhm = np.concatenate(predictive_member_ws, axis=0)
        ensemble = np.sort(member_nhm, axis=-1).transpose(2, 0, 1)
        observed = mask_all > 0.5
        members = ensemble.shape[0]
        first_term = np.mean(np.abs(ensemble - y_all[None, ...]), axis=0)
        coefficients = (
            2.0 * np.arange(1, members + 1) - members - 1.0
        ).astype(np.float32)
        second_term = np.sum(
            coefficients[:, None, None] * ensemble, axis=0,
        ) / float(members ** 2)
        crps = first_term - second_term
        crps_std = float(np.mean(crps[observed]))
        adjustment = float(
            getattr(model, "quantile_interval_adjustment", torch.tensor(0.0))
            .detach().cpu().item()
        )
        lower = ensemble[0] - adjustment
        upper = ensemble[-1] + adjustment
        result.update({
            "quantile_crps_standardized": crps_std,
            "quantile_crps_physical": crps_std * float(stats["target_sd"]),
            "crps_skill_vs_point_forecast_pct": 100.0 * (
                result["mae_standardized"] - crps_std
            ) / max(result["mae_standardized"], 1e-8),
            "prediction_interval_90_coverage": float(np.mean(
                ((y_all >= lower) & (y_all <= upper))[observed]
            )),
            "prediction_interval_90_width_physical": float(np.mean(
                (upper - lower)[observed]
            )) * float(stats["target_sd"]),
        })
        ensemble_phys = (
            ensemble * float(stats["target_sd"]) + float(stats["target_mu"])
        )
        ensemble_phys = np.clip(ensemble_phys, 0.0, float(stats["capacity"]))
        ensemble_std = (
            ensemble_phys - float(stats["target_mu"])
        ) / max(float(stats["target_sd"]), 1e-8)
        first_term = np.mean(np.abs(ensemble_std - y_all[None, ...]), axis=0)
        second_term = np.sum(
            coefficients[:, None, None] * ensemble_std, axis=0,
        ) / float(members ** 2)
        physical_crps = first_term - second_term
        physical_crps_std = float(np.mean(physical_crps[observed]))
        lower_phys = np.clip(
            lower * float(stats["target_sd"]) + float(stats["target_mu"]),
            0.0, float(stats["capacity"]),
        )
        upper_phys = np.clip(
            upper * float(stats["target_sd"]) + float(stats["target_mu"]),
            0.0, float(stats["capacity"]),
        )
        y_phys = y_all * float(stats["target_sd"]) + float(stats["target_mu"])
        result.update({
            "physical_ensemble_members": int(members),
            "physical_ensemble_crps_standardized": physical_crps_std,
            "physical_ensemble_crps_physical": physical_crps_std * float(stats["target_sd"]),
            "physical_ensemble_crps_skill_vs_point_pct": 100.0 * (
                result["mae_standardized"] - physical_crps_std
            ) / max(result["mae_standardized"], 1e-8),
            "physical_ensemble_interval_90_coverage": float(np.mean(
                ((y_phys >= lower_phys) & (y_phys <= upper_phys))[observed]
            )),
            "physical_ensemble_interval_90_width_physical": float(np.mean(
                (upper_phys - lower_phys)[observed]
            )),
        })
    if scale_ws:
        result["scale_weights"] = np.concatenate(scale_ws).mean(axis=0).tolist()
    if tf_ws:
        tf = np.concatenate(tf_ws)
        result["time_frequency_weights"] = tf.mean(axis=0).tolist()
        result["time_frequency_weight_std"] = tf.std(axis=0).tolist()
        tf_prob = tf / np.maximum(tf.sum(axis=1, keepdims=True), 1e-8)
        result["time_frequency_entropy"] = float(
            np.mean(-np.sum(tf_prob * np.log(np.maximum(tf_prob, 1e-8)), axis=1)
                    / np.log(tf_prob.shape[1]))
        )
    if correction_ws:
        corr = np.concatenate(correction_ws)
        result["output_correction_by_horizon"] = corr.mean(axis=0).tolist()
        result["output_correction_mean"] = float(corr.mean())
    if spectral_ws:
        band = np.concatenate(spectral_ws)
        thirds = np.array_split(band, 3, axis=1)
        result["spectral_attention_low_mid_high"] = [float(part.mean()) for part in thirds]
        result["spectral_attention_mean"] = float(band.mean())
        if band.shape[1] == 3:
            result["spectral_attention_std"] = band.std(axis=0).tolist()
            band_prob = band / np.maximum(band.sum(axis=1, keepdims=True), 1e-8)
            result["spectral_attention_entropy"] = float(
                np.mean(-np.sum(band_prob * np.log(np.maximum(band_prob, 1e-8)), axis=1)
                        / np.log(3.0))
            )
    if spectral_residual_gains:
        result["spectral_residual_gain"] = float(
            np.mean(spectral_residual_gains)
        )
    if spectral_raw_ratios:
        values = np.concatenate(spectral_raw_ratios, axis=0)
        result["spectral_raw_contribution_ratio"] = float(values.mean())
        result["spectral_raw_contribution_ratio_std"] = float(values.std())
    if spectral_ratios:
        values = np.concatenate(spectral_ratios, axis=0)
        result["spectral_contribution_ratio"] = float(values.mean())
        result["spectral_contribution_ratio_std"] = float(values.std())
    if frequency_v2_gates:
        gate = np.concatenate(frequency_v2_gates, axis=0)
        result["frequency_v2_gate_low_mid_high"] = gate.mean(axis=0).tolist()
        result["frequency_v2_gate_std"] = gate.std(axis=0).tolist()
    if frequency_v2_scales:
        result["frequency_v2_branch_scale"] = np.stack(
            frequency_v2_scales, axis=0,
        ).mean(axis=0).tolist()
    if frequency_v2_ratios:
        ratio = np.concatenate(frequency_v2_ratios, axis=0)
        result["frequency_v2_contribution_ratio"] = ratio.mean(axis=0).tolist()
        result["frequency_v2_contribution_ratio_std"] = ratio.std(axis=0).tolist()
    if frequency_v2_energies:
        energy = np.concatenate(frequency_v2_energies, axis=0)
        result["frequency_v2_source_energy"] = energy.mean(axis=0).tolist()
    if frequency_v3_gates:
        gate = np.concatenate(frequency_v3_gates, axis=0)
        result["frequency_v3_gate_mean"] = float(gate.mean())
        result["frequency_v3_gate_std"] = float(gate.std())
    if frequency_v3_scales:
        result["frequency_v3_residual_scale"] = float(np.mean(frequency_v3_scales))
    if frequency_v3_ratios:
        ratio = np.concatenate(frequency_v3_ratios, axis=0)
        result["frequency_v3_contribution_ratio"] = float(ratio.mean())
        result["frequency_v3_contribution_ratio_std"] = float(ratio.std())
    if frequency_v3_energies:
        result["frequency_v3_spectral_energy"] = float(
            np.concatenate(frequency_v3_energies, axis=0).mean()
        )
    if frequency_v4_gates:
        gate = np.concatenate(frequency_v4_gates, axis=0)
        result["frequency_v4_branch_gate_global_local"] = gate.mean(axis=0).tolist()
        result["frequency_v4_branch_gate_std"] = gate.std(axis=0).tolist()
    if frequency_v4_scales:
        result["frequency_v4_branch_scale_global_local"] = np.stack(
            frequency_v4_scales, axis=0,
        ).mean(axis=0).tolist()
    if frequency_v4_ratios:
        ratio = np.concatenate(frequency_v4_ratios, axis=0)
        result["frequency_v4_contribution_ratio_global_local"] = ratio.mean(axis=0).tolist()
        result["frequency_v4_contribution_ratio_std"] = ratio.std(axis=0).tolist()
    if frequency_v4_energies:
        energy = np.concatenate(frequency_v4_energies, axis=0)
        result["frequency_v4_source_energy_global_local"] = energy.mean(axis=0).tolist()
    if frequency_v4_local_gates:
        local_gate = np.concatenate(frequency_v4_local_gates, axis=0)
        result["frequency_v4_local_spectral_gate_mean"] = float(local_gate.mean())
        result["frequency_v4_local_spectral_gate_std"] = float(local_gate.std())
    if frequency_v5_nulls:
        null_probability = np.concatenate(frequency_v5_nulls, axis=0)
        result["frequency_v5_null_probability_mean"] = float(null_probability.mean())
        result["frequency_v5_null_probability_std"] = float(null_probability.std())
    if frequency_v5_channel_gates:
        channel_gate = np.concatenate(frequency_v5_channel_gates, axis=0)
        result["frequency_v5_channel_gate_mean"] = float(channel_gate.mean())
        result["frequency_v5_channel_gate_std"] = float(channel_gate.std())
    if frequency_v5_scales:
        result["frequency_v5_residual_scale"] = float(np.mean(frequency_v5_scales))
    if frequency_v5_ratios:
        ratio = np.concatenate(frequency_v5_ratios, axis=0)
        result["frequency_v5_contribution_ratio"] = float(ratio.mean())
        result["frequency_v5_contribution_ratio_std"] = float(ratio.std())
    if frequency_v6_gates:
        gate = np.concatenate(frequency_v6_gates, axis=0)
        result["frequency_v6_band_gate_low_mid_high"] = gate.mean(axis=0).tolist()
        result["frequency_v6_band_gate_std"] = gate.std(axis=0).tolist()
    if frequency_v6_energies:
        energy = np.concatenate(frequency_v6_energies, axis=0)
        result["frequency_v6_band_energy_fraction"] = energy.mean(axis=0).tolist()
    if frequency_v6_scales:
        result["frequency_v6_residual_scale"] = float(np.mean(frequency_v6_scales))
    if frequency_v6_ratios:
        ratio = np.concatenate(frequency_v6_ratios, axis=0)
        result["frequency_v6_contribution_ratio"] = float(ratio.mean())
        result["frequency_v6_contribution_ratio_std"] = float(ratio.std())
    if frequency_v6_magnitude:
        value = np.concatenate(frequency_v6_magnitude, axis=0)
        result["frequency_v6_magnitude_adjustment"] = float(value.mean())
    if frequency_v6_phase:
        value = np.concatenate(frequency_v6_phase, axis=0)
        result["frequency_v6_phase_adjustment"] = float(value.mean())
    if frequency_v7_gates:
        value = np.concatenate(frequency_v7_gates, axis=0)
        result["frequency_v7_band_gate_mid_high"] = value.mean(axis=0).tolist()
        result["frequency_v7_band_gate_std"] = value.std(axis=0).tolist()
    if frequency_v7_coherences:
        value = np.concatenate(frequency_v7_coherences, axis=0)
        result["frequency_v7_coherence_mid_high"] = value.mean(axis=0).tolist()
    if frequency_v7_stabilities:
        value = np.concatenate(frequency_v7_stabilities, axis=0)
        result["frequency_v7_stability_mid_high"] = value.mean(axis=0).tolist()
    if frequency_v7_priors:
        value = np.stack(frequency_v7_priors, axis=0)
        result["frequency_v7_horizon_prior_mid_high"] = value.mean(axis=0).tolist()
    if frequency_v7_clips:
        value = np.concatenate(frequency_v7_clips, axis=0)
        result["frequency_v7_clip_factor_mean"] = float(value.mean())
        result["frequency_v7_clip_fraction"] = float((value < 0.999999).mean())
    if frequency_v7_ratios:
        value = np.concatenate(frequency_v7_ratios, axis=0)
        result["frequency_v7_contribution_ratio"] = float(value.mean())
        result["frequency_v7_contribution_ratio_std"] = float(value.std())
        result["frequency_v7_contribution_ratio_quantiles"] = {
            "p50": float(np.quantile(value, 0.50)),
            "p90": float(np.quantile(value, 0.90)),
            "p95": float(np.quantile(value, 0.95)),
            "p99": float(np.quantile(value, 0.99)),
        }
    if frequency_v7_magnitude:
        value = np.concatenate(frequency_v7_magnitude, axis=0)
        result["frequency_v7_magnitude_adjustment"] = float(value.mean())
    if frequency_v7_phase:
        value = np.concatenate(frequency_v7_phase, axis=0)
        result["frequency_v7_phase_adjustment"] = float(value.mean())
    if frequency_v8_gates:
        value = np.concatenate(frequency_v8_gates, axis=0)
        result["frequency_v8_band_channel_gate"] = value.mean(axis=0).tolist()
        result["frequency_v8_band_channel_gate_std"] = value.std(axis=0).tolist()
    if frequency_v8_energies:
        value = np.concatenate(frequency_v8_energies, axis=0)
        result["frequency_v8_source_energy_fraction"] = value.mean(axis=0).tolist()
    if frequency_v8_robust_clips:
        value = np.concatenate(frequency_v8_robust_clips, axis=0)
        result["frequency_v8_robust_input_clip_fraction"] = float(value.mean())
    if frequency_v8_caps:
        value = np.stack(frequency_v8_caps, axis=0)
        result["frequency_v8_residual_cap_by_channel"] = value.mean(axis=0).tolist()
    if frequency_v8_residual_clips:
        value = np.concatenate(frequency_v8_residual_clips, axis=0)
        result["frequency_v8_residual_clip_fraction"] = float(value.mean())
    if frequency_v8_raw_ratios:
        value = np.concatenate(frequency_v8_raw_ratios, axis=0)
        result["frequency_v8_raw_contribution_ratio"] = float(value.mean())
    if frequency_v8_ratios:
        value = np.concatenate(frequency_v8_ratios, axis=0)
        result["frequency_v8_contribution_ratio"] = float(value.mean())
        result["frequency_v8_contribution_ratio_std"] = float(value.std())
        result["frequency_v8_contribution_ratio_quantiles"] = {
            "p50": float(np.quantile(value, 0.50)),
            "p90": float(np.quantile(value, 0.90)),
            "p95": float(np.quantile(value, 0.95)),
            "p99": float(np.quantile(value, 0.99)),
        }
    if frequency_v8_channel_ratios:
        value = np.concatenate(frequency_v8_channel_ratios, axis=0)
        result["frequency_v8_channel_contribution_ratio"] = value.mean(axis=0).tolist()
    if frequency_v9_trend_mix:
        value = np.concatenate(frequency_v9_trend_mix, axis=0)
        result["frequency_v9_trend_mix"] = float(value.mean())
        result["frequency_v9_trend_mix_std"] = float(value.std())
    if frequency_v9_gates:
        value = np.concatenate(frequency_v9_gates, axis=0)
        result["frequency_v9_band_gate"] = value.mean(axis=0).tolist()
        result["frequency_v9_band_gate_std"] = value.std(axis=0).tolist()
    if frequency_v9_scales:
        value = np.concatenate(frequency_v9_scales, axis=0)
        result["frequency_v9_residual_scale"] = float(value.mean())
        result["frequency_v9_residual_scale_std"] = float(value.std())
        result["frequency_v9_residual_scale_quantiles"] = {
            "p10": float(np.quantile(value, 0.10)),
            "p50": float(np.quantile(value, 0.50)),
            "p90": float(np.quantile(value, 0.90)),
        }
    if frequency_v9_trend_energy:
        value = np.concatenate(frequency_v9_trend_energy, axis=0)
        result["frequency_v9_trend_residual_energy_fraction"] = value.mean(axis=0).tolist()
    if frequency_v9_raw_ratios:
        value = np.concatenate(frequency_v9_raw_ratios, axis=0)
        result["frequency_v9_raw_contribution_ratio"] = float(value.mean())
    if frequency_v9_ratios:
        value = np.concatenate(frequency_v9_ratios, axis=0)
        result["frequency_v9_contribution_ratio"] = float(value.mean())
        result["frequency_v9_contribution_ratio_std"] = float(value.std())
        result["frequency_v9_contribution_ratio_quantiles"] = {
            "p10": float(np.quantile(value, 0.10)),
            "p50": float(np.quantile(value, 0.50)),
            "p90": float(np.quantile(value, 0.90)),
        }
    if frequency_v9_target_ratios:
        value = np.concatenate(frequency_v9_target_ratios, axis=0)
        result["frequency_v9_target_contribution_ratio"] = float(value.mean())
    if frequency_v9_robust_clips:
        value = np.concatenate(frequency_v9_robust_clips, axis=0)
        result["frequency_v9_robust_input_clip_fraction"] = float(value.mean())
    if semantic_ws:
        semantic = np.concatenate(semantic_ws)
        result["semantic_residual_gate_mean"] = float(semantic.mean())
        result["semantic_residual_gate_std"] = float(semantic.std())
    if semantic_head_ws:
        semantic_heads = np.concatenate(semantic_head_ws)
        result["semantic_cma_head_weights"] = semantic_heads.mean(axis=0).tolist()
        result["semantic_cma_head_weight_std"] = semantic_heads.std(axis=0).tolist()
    if scale_prompt_ws:
        prompt_gate = np.concatenate(scale_prompt_ws)
        result["scale_prompt_gate_by_scale"] = (
            prompt_gate.mean(axis=(0, 1, 3)).tolist()
        )
        result["scale_prompt_gate_mean"] = float(prompt_gate.mean())
    if scale_prompt_head_ws:
        prompt_heads = np.concatenate(scale_prompt_head_ws)
        result["scale_prompt_head_weights"] = (
            prompt_heads.mean(axis=(0, 1, 2)).tolist()
        )
        result["scale_prompt_head_weight_std"] = (
            prompt_heads.std(axis=(0, 1, 2)).tolist()
        )
    if scale_prompt_match:
        result["scale_prompt_match_accuracy"] = float(np.mean(scale_prompt_match))
    if fsra_ws:
        fsra = np.concatenate(fsra_ws)
        result["fsra_residual_gate_mean"] = float(fsra.mean())
        result["fsra_residual_gate_std"] = float(fsra.std())
    if fsra_head_ws:
        fsra_heads = np.concatenate(fsra_head_ws)
        reduce_axes = tuple(range(fsra_heads.ndim - 1))
        result["fsra_cma_head_weights"] = fsra_heads.mean(axis=reduce_axes).tolist()
        result["fsra_cma_head_weight_std"] = fsra_heads.std(axis=reduce_axes).tolist()
        result["fsra_cma_head_entropy"] = float(
            np.mean(-np.sum(
                fsra_heads * np.log(np.maximum(fsra_heads, 1e-8)), axis=-1,
            ) / np.log(fsra_heads.shape[-1]))
        )
    relation_scale = getattr(model, "last_fsra_relation_scale", None)
    if relation_scale is not None:
        result["fsra_relation_scale"] = float(relation_scale.cpu())
    reference_weights = getattr(model, "last_reference_weights", None)
    if reference_weights is not None:
        result["reference_weights_solar_persistence_daily"] = (
            reference_weights.detach().cpu().tolist()
        )
    if analyze_regimes and xs and scale_ws:
        result["regime_breakdown"] = _regime_breakdown(
            np.concatenate(xs), y_all, p_all, mask_all, stats, scale_ws, tf_ws,
            correction_ws, spectral_ws)
    return result


@torch.no_grad()
def evaluate_mc_dropout(model, loader, device, stats, members=19,
                        calibration=1.0, seed=2026):
    """Reproducible MC-Dropout ensemble with optional validation dilation."""
    predictions = []
    y_all = mask_all = None
    for member in range(members):
        torch.manual_seed(int(seed + member))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed + member))
        model.eval()
        for module in model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
                module.train()
        member_pred = []
        ys = []
        masks = []
        for batch in loader:
            x, y, y_mask, cycle, future_x, prompt_x = unpack_batch(batch, device)
            pred = predict(model, x, cycle, future_x, prompt_x)
            member_pred.append(pred.cpu().numpy())
            if member == 0:
                ys.append(y.cpu().numpy())
                masks.append(y_mask.cpu().numpy())
        predictions.append(np.concatenate(member_pred))
        if member == 0:
            y_all = np.concatenate(ys)
            mask_all = np.concatenate(masks)
    ensemble = np.stack(predictions, axis=0)
    center = ensemble.mean(axis=0, keepdims=True)
    ensemble = center + float(calibration) * (ensemble - center)
    ensemble = np.sort(ensemble, axis=0)
    observed = mask_all > 0.5
    first_term = np.mean(np.abs(ensemble - y_all[None, ...]), axis=0)
    coefficients = (
        2.0 * np.arange(1, members + 1) - members - 1.0
    ).astype(np.float32)
    second_term = np.sum(
        coefficients[:, None, None] * ensemble, axis=0,
    ) / float(members ** 2)
    crps = first_term - second_term
    crps_std = float(np.mean(crps[observed]))
    lower, upper = np.quantile(ensemble, [0.05, 0.95], axis=0)
    ensemble_phys = (
        ensemble * float(stats["target_sd"]) + float(stats["target_mu"])
    )
    ensemble_phys = np.clip(ensemble_phys, 0.0, float(stats["capacity"]))
    ensemble_std = (
        ensemble_phys - float(stats["target_mu"])
    ) / max(float(stats["target_sd"]), 1e-8)
    first_term_phys = np.mean(
        np.abs(ensemble_std - y_all[None, ...]), axis=0,
    )
    second_term_phys = np.sum(
        coefficients[:, None, None] * ensemble_std, axis=0,
    ) / float(members ** 2)
    physical_crps_std = float(np.mean(
        (first_term_phys - second_term_phys)[observed]
    ))
    lower_phys, upper_phys = np.quantile(ensemble_phys, [0.05, 0.95], axis=0)
    y_phys = y_all * float(stats["target_sd"]) + float(stats["target_mu"])
    point_mae = float(np.mean(np.abs(center[0] - y_all)[observed]))
    return {
        "mc_dropout_members": int(members),
        "mc_dropout_calibration": float(calibration),
        "mc_dropout_crps_standardized": crps_std,
        "mc_dropout_crps_physical": crps_std * float(stats["target_sd"]),
        "crps_skill_vs_point_forecast_pct": 100.0 * (
            point_mae - crps_std
        ) / max(point_mae, 1e-8),
        "prediction_interval_90_coverage": float(np.mean(
            ((y_all >= lower) & (y_all <= upper))[observed]
        )),
        "prediction_interval_90_width_physical": float(np.mean(
            (upper - lower)[observed]
        )) * float(stats["target_sd"]),
        "physical_ensemble_members": int(members),
        "physical_ensemble_crps_standardized": physical_crps_std,
        "physical_ensemble_crps_physical": physical_crps_std * float(stats["target_sd"]),
        "physical_ensemble_crps_skill_vs_point_pct": 100.0 * (
            point_mae - physical_crps_std
        ) / max(point_mae, 1e-8),
        "physical_ensemble_interval_90_coverage": float(np.mean(
            ((y_phys >= lower_phys) & (y_phys <= upper_phys))[observed]
        )),
        "physical_ensemble_interval_90_width_physical": float(np.mean(
            (upper_phys - lower_phys)[observed]
        )),
        "mc_dropout_scale_mean_standardized": float(np.std(ensemble, axis=0)[observed].mean()),
    }


@torch.no_grad()
def _validation_mc_dropout_scale(model, loader, device, seed=2026, members=19):
    """Validation-only dilation targeting marginal 90% MC-Dropout coverage."""
    predictions = []
    y_all = mask_all = None
    for member in range(members):
        torch.manual_seed(int(seed + member))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed + member))
        model.eval()
        for module in model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
                module.train()
        ps = []
        ys = []
        masks = []
        for batch in loader:
            x, y, y_mask, cycle, future_x, prompt_x = unpack_batch(batch, device)
            ps.append(predict(model, x, cycle, future_x, prompt_x).cpu().numpy())
            if member == 0:
                ys.append(y.cpu().numpy())
                masks.append(y_mask.cpu().numpy())
        predictions.append(np.concatenate(ps))
        if member == 0:
            y_all = np.concatenate(ys)
            mask_all = np.concatenate(masks)
    ensemble = np.stack(predictions, axis=0)
    center = ensemble.mean(axis=0)
    spread = ensemble.std(axis=0).clip(1e-4)
    values = (np.abs(y_all - center) / spread)[mask_all > 0.5]
    try:
        quantile = float(np.quantile(values, 0.90, method="higher"))
    except TypeError:
        quantile = float(np.quantile(values, 0.90, interpolation="higher"))
    # 19 stochastic members can be severely under-dispersed; cap the
    # validation correction to keep pathological intervals auditable.
    return float(np.clip(quantile / 1.6448536269514722, 0.25, 10.0))


def _compact_counterfactual(result):
    compact = {key: result[key] for key in (
        "mse_standardized", "nmae_capacity", "ramp_mae_norm"
    )}
    regimes = {}
    for indicator, content in (result.get("regime_breakdown") or {}).items():
        regimes[indicator] = {
            label: {
                "mse_standardized": row["mse_standardized"],
                "nmae_capacity": row["nmae_capacity"],
            }
            for label, row in content["groups"].items()
        }
    compact["regime_errors"] = regimes
    return compact


def calibrate_reference_ensemble(args, loaders, stats, device):
    """Fit only the three reference weights, using train/validation data.

    The returned logits are frozen before the residual model is optimized,
    removing the base/residual identifiability shortcut from the experiment.
    Test targets are never used for this calibration.
    """
    reference = PhysicsReferenceEnsemble(stats, args.horizon, args.cycle_len).to(device)
    opt = torch.optim.Adam([reference.reference_logits], lr=5e-2)
    best_mse, best_logits, patience_left = float("inf"), None, 4
    for _ in range(20):
        reference.train()
        for batch in loaders[0]:
            xb, yb, ymb, cb, fxb, _ = unpack_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            pred = reference(xb, cb, fxb)
            loss = _masked_mean((pred - yb).square(), ymb)
            loss.backward()
            opt.step()
        val = evaluate(reference, loaders[1], device, stats)
        if val["mse_standardized"] < best_mse - 1e-8:
            best_mse = val["mse_standardized"]
            best_logits = reference.reference_logits.detach().clone()
            patience_left = 4
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    if best_logits is None:
        raise RuntimeError("reference calibration did not produce a checkpoint")
    return best_logits


def _matched_random_prompt_bank(prompt_by_start):
    """Replace each frozen GPT token prototype by a deterministic random one.

    Identical text embeddings map to the same random prototype, so this control
    preserves categorical information and adapter capacity while removing the
    geometry learned during language pre-training.  Per-token mean and standard
    deviation are matched before the same training-only centering used by the
    correct-prompt condition.
    """
    bank = {}
    randomized = {}
    for start, value in prompt_by_start.items():
        value = value.astype(np.float32)
        tokens = []
        for token in value:
            key = token.astype(np.float16).tobytes()
            if key not in bank:
                seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
                rng = np.random.default_rng(seed)
                sample = rng.standard_normal(token.shape).astype(np.float32)
                sample = ((sample - sample.mean()) /
                          max(float(sample.std()), 1e-6))
                sample = sample * max(float(token.std()), 1e-6) + float(token.mean())
                bank[key] = sample.astype(np.float32)
            tokens.append(bank[key])
        randomized[start] = np.stack(tokens).astype(np.float32)
    return randomized


def _all_but_top_prompt_bank(prompt_by_start, train_starts,
                             components=2, seed=2026):
    """Training-only GPT-2 common-direction removal and unit normalization.

    GPT-2 hidden states are anisotropic.  A randomized two-vector power
    iteration avoids a heavyweight PCA dependency while fitting only on
    training prompt tokens.  The frozen mean/components are then applied to
    every split; validation/test never influence the transform.
    """
    train = np.concatenate(
        [prompt_by_start[int(start)].astype(np.float32) for start in train_starts],
        axis=0,
    )
    center = train.mean(axis=0, keepdims=True)
    centered = train - center
    k = min(int(components), centered.shape[1], max(0, centered.shape[0] - 1))
    if k:
        rng = np.random.default_rng(seed)
        basis = rng.standard_normal((centered.shape[1], k)).astype(np.float32)
        basis, _ = np.linalg.qr(basis)
        for _ in range(6):
            basis = centered.T @ (centered @ basis)
            basis, _ = np.linalg.qr(basis)
        components_matrix = basis.T.astype(np.float32)
    else:
        components_matrix = np.empty((0, centered.shape[1]), dtype=np.float32)

    transformed = {}
    for start, value in prompt_by_start.items():
        token = value.astype(np.float32) - center
        if k:
            token = token - (token @ components_matrix.T) @ components_matrix
        norm = np.linalg.norm(token, axis=-1, keepdims=True)
        transformed[start] = (token / np.maximum(norm, 1e-6)).astype(np.float32)
    explained = []
    total_energy = float(np.square(centered).sum())
    if k and total_energy > 0.0:
        for direction in components_matrix:
            explained.append(float(np.square(centered @ direction).sum() / total_energy))
    return transformed, {
        "fit_split": "training_only",
        "removed_components": int(k),
        "removed_energy_fraction": explained,
        "token_l2_normalized": True,
    }


def _fsra_adapter_state(model):
    """Return only site-independent adapter parameters and buffers."""
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if key.startswith("fsra_") and key != "fsra_channel_embeddings"
    }


def _load_fsra_adapter(model, checkpoint_path):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    incoming = payload.get("adapter_state", payload)
    current = model.state_dict()
    bad = {
        key: (tuple(value.shape), tuple(current[key].shape) if key in current else None)
        for key, value in incoming.items()
        if key not in current or tuple(value.shape) != tuple(current[key].shape)
    }
    if bad:
        raise ValueError(f"incompatible FSRA transfer checkpoint: {bad}")
    model.load_state_dict(incoming, strict=False)
    return payload


def _save_fsra_adapter(model, checkpoint_path, metadata):
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "adapter_state": _fsra_adapter_state(model),
        "metadata": dict(metadata),
    }, path)


def adapter_stage_policy(
    *,
    skip_adapter_stages: bool,
    fsra_requested: bool,
    prompt_requested: bool,
) -> tuple[bool, bool]:
    """Resolve FSRA and Prompt/CMA fitting for one experiment stage."""
    enabled = not bool(skip_adapter_stages)
    return bool(fsra_requested and enabled), bool(prompt_requested and enabled)


_RETIRED_OURS_RESULT_FIELDS = frozenset({
    "frequency_v2_mode", "frequency_v3_mode", "frequency_v4_mode",
    "frequency_v5_mode", "frequency_v6_mode", "frequency_v7_mode",
    "frequency_v8_mode", "frequency_v9_mode", "spectral_mkan_mode",
    "clean_frequency_ablation", "candidate_auxiliary", "candidate_loss_weight",
    "output_correction_floor", "output_router_bias", "anchor_output_mode",
    "output_correction_scale", "local_token_refiner", "local_refiner_dropout",
    "scale_token_mixer", "scale_token_mixer_adaptive",
    "scale_token_mixer_dropout", "scale_token_branch_weights",
    "scale_token_residual_scale", "fsra_mae_weight", "router_activation",
    "expert_drop_path", "gtr_dropout", "model_variant", "spectral_features",
    "counterfactuals", "counterfactual_split", "fsra_counterfactuals",
    "fsra_paired_backbone_test", "fsra_paired_backbone_val",
    "reference_paired_backbone_test", "reference_paired_backbone_val",
    "reference_strength", "reference_residual_active",
    "reference_path_uses_future_solar_geometry", "fsra_strength",
    "fsra_control", "fsra_descriptions", "fsra_role_names",
    "fsra_transfer_loaded_from", "fsra_transfer_source_metadata",
    "fsra_target_semantic_training", "probabilistic_output",
    "probabilistic_state_conditioning", "uncertainty_validation_calibration",
    "quantile_output", "quantile_state_conditioning",
    "quantile_interval_adjustment_standardized", "protocol_note",
})


def clean_current_ours_result(payload: dict) -> dict:
    """Remove inactive legacy-branch metadata from the public Ours result."""
    if payload.get("model") != "ours":
        return payload
    return {
        key: value
        for key, value in payload.items()
        if key not in _RETIRED_OURS_RESULT_FIELDS
    }


def run_one(name, args, data, device):
    x, tr, va, te, cycle_ids, stats = data
    cma_model = name == "ours" or name.startswith("pcarr_v23_")
    fsra_model = (name.startswith("pcarr_v14_fsra")
                  or name.startswith("pcarr_v16_fsra")
                  or name.startswith("pcarr_v17_fsra")
                  or name.startswith("pcarr_v18_fsra")
                  or name.startswith("pcarr_v19_fsra")
                  or name.startswith("pcarr_v21_fsra")
                  or name.startswith("pcarr_v22_fsra")
                  or name.startswith("pcarr_v23_fsra"))
    scale_prompt_model = (name.startswith("pcarr_v15_gpt2_ms")
                          or name.startswith("pcarr_v16_fsra")
                          or name.startswith("pcarr_v17_fsra")
                          or name.startswith("pcarr_v18_fsra")
                          or name.startswith("pcarr_v19_fsra")
                          or "v21_fsra" in name and "gpt2proto" in name
                          or "v22_fsra" in name and "gpt2semantic" in name)
    reference_residual_model = (
        name.startswith("pcarr_v20_refrel")
        or name.startswith("pcarr_v21_fsra")
        or name.startswith("pcarr_v22_fsra")
        or name.startswith("pcarr_v23_fsra")
    )
    prompt_shuffle = name.endswith("_shuffle") or "promptshuffle" in name
    prompt_random = name.endswith("_random") or "promptrandom" in name
    prompt_stats = "statsmlp" in name
    # V14 text is static channel metadata, not a sample prompt.  Encoding is
    # performed offline and cached; the forecasting graph contains no language
    # model and receives no future observations or test statistics.
    if fsra_model:
        if "_roles" in name:
            frozen_vectors, descriptions, role_names = load_channel_role_signatures(
                stats["feature_names"], ROOT / "external" / "hf_cache",
            )
            stats["_fsra_role_names"] = role_names
        else:
            frozen_vectors, descriptions = load_channel_semantics(
                stats["feature_names"], ROOT / "external" / "hf_cache",
            )
            stats.pop("_fsra_role_names", None)
        if name.endswith("_shuffle"):
            control = "shuffle"
        elif name.endswith("_zero") or "_fsrazero_" in name:
            control = "zero"
        elif name.endswith("_random"):
            control = "random"
        elif name.endswith("_id"):
            control = "learned_id"
        else:
            control = "correct"
        stats["_fsra_vectors"] = torch.tensor(
            controlled_channel_semantics(frozen_vectors, control, args.seed),
            dtype=torch.float32,
        )
        stats["_fsra_control"] = control
        stats["_fsra_descriptions"] = descriptions
    else:
        stats.pop("_fsra_vectors", None)
        stats.pop("_fsra_control", None)
        stats.pop("_fsra_descriptions", None)
        stats.pop("_fsra_role_names", None)
    future_models = {"nwp_transformer", "vanilla_history_nwpdecoder"}
    if name == "crossunet" and bool(stats.get("future_indices")):
        future_models.add("crossunet")
    solar_future_model = name in {
        "smart_persistence", "reference_ensemble", "reference_ensemble_probglobal",
    } or (
        (name.startswith("psrc_v2") or name.startswith("pcatfr") or name.startswith("pcarr"))
        and not name.endswith("plain_anchor") and args.anchor_output_mode != "direct"
        and (not name.startswith("pcarr_v23_") or args.ours_solar_anchor)
    )
    model_uses_future = (name.endswith("_future") or "nwpdecoder" in name
                         or name in future_models or solar_future_model)
    projection_uses_future = args.evaluation_projection == "nonnegative_night"
    include_future = model_uses_future or projection_uses_future
    # M21 is retained as the explicitly legacy oracle-weather path. Every new
    # future-conditioned model receives only the provenance-audited tensor.
    if solar_future_model or projection_uses_future:
        audited_future = stats.get("solar_future_indices")
        if not audited_future:
            raise ValueError(
                "solar trajectory or night projection requires audited "
                "deterministic future solar features"
            )
    else:
        audited_future = (None if name.endswith("_future")
                          else stats.get("future_indices") if include_future else None)
    prompt_by_start = None
    prompt_preprocessing_metadata = None
    prompt_model = (name.startswith("pcarr_v4_") or name.startswith("pcarr_v5_")
                    or name.startswith("pcarr_v6_") or name.startswith("pcarr_v7_")
                    or name.startswith("pcarr_v8_") or name.startswith("pcarr_v9_")
                    or name.startswith("pcarr_v10_") or name.startswith("pcarr_v11_")
                    or name.startswith("pcarr_v12_") or name.startswith("pcarr_v13_")
                    or cma_model
                    or scale_prompt_model)
    prompt_uses_future_solar = bool(
        prompt_model
        and args.prompt_future_solar_policy == "deterministic_solar_geometry"
    )
    if prompt_model:
        stats["seq_len"] = args.seq_len
        # A validation-only screen must not even pre-encode prompts for the
        # outer test windows.  Only training and validation prompt keys are
        # needed because the test loader is never iterated in this mode.
        all_starts = np.concatenate((tr, va)) if args.validation_only else np.concatenate((tr, va, te))
        high_information_prompt = (
            cma_model
            and args.ours_cma_prompt_features in {
                "gpt2_sequence", "random_gpt2_sequence", "numeric_sequence",
            }
        )
        prompt_version = (10 if high_information_prompt else
                          9 if cma_model else
                          8 if (name.startswith("pcarr_v22_fsra")
                                and not name.endswith("_a")) else
                          5 if (name.startswith("pcarr_v21_fsra")
                                or name.startswith("pcarr_v22_fsra")) else
                          7 if (name.startswith("pcarr_v18_fsra")
                                or name.startswith("pcarr_v19_fsra")) else
                          6 if scale_prompt_model else
                          5 if name.startswith("pcarr_v13_") else
                          4 if (name.startswith("pcarr_v10_") or name.startswith("pcarr_v11_")) else
                          3 if (name.startswith("pcarr_v7_") or name.startswith("pcarr_v8_")
                                or name.startswith("pcarr_v9_")) else
                          2 if (name.startswith("pcarr_v5_") or name.startswith("pcarr_v6_")
                                or name.startswith("pcarr_v12_")) else 1)
        if prompt_version >= 8 and prompt_version != 9:
            # Fit every categorical boundary on the selected training windows
            # only, then freeze it before any held-out prompt is created.
            stats["prompt_regime_thresholds"] = fit_prompt_regime_thresholds(
                x, tr, stats,
            )
        if prompt_version == 10:
            cache_dir = {
                "gpt2_sequence": "prompt_cache_v10_sequence_gpt2_lasttoken",
                "random_gpt2_sequence": "prompt_cache_v10_sequence_random_gpt2_lasttoken",
                "numeric_sequence": "prompt_cache_v10_sequence_numeric",
            }[args.ours_cma_prompt_features]
        else:
            cache_dir = ("prompt_cache_v9_variable_regime_gpt2_lasttoken" if prompt_version == 9 else
                     "prompt_cache_v8_quantile_gpt2" if prompt_version >= 8 else
                     "prompt_cache_v7_qwen" if prompt_version == 7 else
                     "prompt_cache_v6" if prompt_version == 6 else
                     "prompt_cache_v5" if prompt_version == 5 else
                     "prompt_cache_v4" if prompt_version == 4 else
                     "prompt_cache_v3_t3time_variable" if prompt_version == 3 else
                     "prompt_cache_v2" if prompt_version == 2 else "prompt_cache")
        threshold_tag = ""
        if prompt_version >= 8 and prompt_version != 9:
            threshold_payload = json.dumps(
                stats["prompt_regime_thresholds"], sort_keys=True,
            ).encode("utf-8")
            threshold_tag = "_" + hashlib.sha256(threshold_payload).hexdigest()[:10]
        future_prompt_tag = "" if prompt_uses_future_solar else "_historyonly"
        cache_file = (ROOT / "experiments" / cache_dir /
                      f"{Path(args.data).stem}_L{args.seq_len}_H{args.horizon}"
                      f"_W{args.max_windows}{threshold_tag}{future_prompt_tag}.npz")
        llm_prompts, numeric_prompts = build_or_load_prompt_features(
            x, all_starts, stats, args.horizon, cache_file,
            ROOT / "external" / "hf_cache", prompt_version,
            include_future_solar=prompt_uses_future_solar,
            encoder_kind=(
                "numeric" if args.ours_cma_prompt_features == "numeric_sequence" else
                "gpt2_random" if args.ours_cma_prompt_features == "random_gpt2_sequence" else
                "gpt2"
            ),
            random_seed=args.seed,
        )
        prompt_by_start = (
            numeric_prompts if (
                name == "pcarr_v4_stats_mlp" or prompt_stats
                or (cma_model
                    and args.ours_cma_prompt_features == "numeric_sequence")
            )
            else llm_prompts
        )
        if (cma_model
                and args.ours_cma_prompt_features == "gpt2_numeric"):
            prompt_by_start = {
                start: np.concatenate(
                    [llm_prompts[start], numeric_prompts[start]], axis=-1,
                ).astype(np.float32)
                for start in prompt_by_start
            }
        if cma_model:
            first_prompt = next(iter(prompt_by_start.values()))
            stats["_dynamic_prompt_dim"] = int(first_prompt.shape[-1])
        if prompt_random:
            prompt_by_start = _matched_random_prompt_bank(prompt_by_start)
        if name.startswith("pcarr_v22_fsra"):
            prompt_by_start, prompt_preprocessing_metadata = (
                _all_but_top_prompt_bank(
                    prompt_by_start, tr, components=2, seed=args.seed,
                )
            )
            prompt_preprocessing_metadata["prompt_version"] = int(prompt_version)
            prompt_preprocessing_metadata["thresholds"] = (
                stats.get("prompt_regime_thresholds")
            )
        elif (name.startswith("pcarr_v12_zerosafe_") or name.startswith("pcarr_v13_")
              or cma_model or scale_prompt_model):
            # Frozen GPT-2 sentence vectors are strongly anisotropic: their
            # shared language direction can be orders of magnitude larger than
            # the between-window operating-state variation.  Remove only the
            # training-set token mean and use a token-wise RMS scale.  This is
            # leakage-safe and makes shuffled-prompt controls probe pairing,
            # rather than merely the presence of a common text embedding.
            train_prompt = np.stack([prompt_by_start[int(start)] for start in tr]).astype(np.float32)
            prompt_center = train_prompt.mean(axis=0)
            centered_train = train_prompt - prompt_center[None, ...]
            token_scale = np.sqrt(np.mean(centered_train ** 2, axis=(0, 2))).reshape(-1, 1)
            token_scale = np.maximum(token_scale, 1e-3)
            prompt_by_start = {
                start: ((value.astype(np.float32) - prompt_center) / token_scale).astype(np.float32)
                for start, value in prompt_by_start.items()
            }
            prompt_preprocessing_metadata = {
                "method": "training_token_center_rms",
                "prompt_version": int(prompt_version),
                "minimum_scale": 1e-3,
                "features": (
                    args.ours_cma_prompt_features
                    if cma_model else "gpt2"
                ),
            }
        if name.endswith("_zero") or "promptzero" in name:
            prompt_by_start = {
                start: np.zeros_like(value) for start, value in prompt_by_start.items()
            }
    fixed_loaders = [
        DataLoader(
            WindowSet(
                x, idx, args.seq_len, args.horizon, args.cycle_len, cycle_ids,
                stats.get("target_observed"), include_future=include_future,
                future_indices=audited_future,
                prompt_by_start=prompt_by_start,
                prompt_shuffle=prompt_shuffle,
            ),
            args.batch, shuffle=(i == 0), num_workers=0,
        )
        for i, idx in enumerate((tr, va, te))
    ]
    semantic_train_loader = fixed_loaders[0]
    loaders = fixed_loaders
    if args.train_sampling_policy == "epoch_uniform":
        epoch_pool = stats.get("_epoch_train_pool")
        if epoch_pool is None or len(epoch_pool) < len(tr):
            raise ValueError("epoch-uniform policy requires the full training pool")
        epoch_dataset = WindowSet(
            x, epoch_pool, args.seq_len, args.horizon, args.cycle_len, cycle_ids,
            stats.get("target_observed"), include_future=include_future,
            future_indices=audited_future,
            # The numerical backbone and FSRA stages do not consume sample
            # prompts.  The fixed semantic loader below retains the audited
            # prompt cache for the later GPT-2 adapter stage.
            prompt_by_start=None,
            prompt_shuffle=False,
        )
        epoch_sampler = EpochStratifiedSampler(
            len(epoch_pool), len(tr), args.seed,
        )
        loaders[0] = DataLoader(
            epoch_dataset, args.batch, sampler=epoch_sampler, num_workers=0,
        )
    # Offline text preparation may instantiate a frozen transformer or execute
    # GPU kernels before the forecasting model is built.  Reset here so a
    # prompt-enabled experiment and its no-prompt control start from exactly
    # the same model and DataLoader RNG state.
    seed_all(args.seed)
    model, loss_kind = build_model(name, args, x.shape[1], stats)
    if args.loss_kind_override is not None:
        loss_kind = args.loss_kind_override
    model = model.to(device)
    if reference_residual_model and args.anchor_output_mode == "direct":
        # Direct mode is the strict history-only interface. Keep the stored
        # state consistent with the executed graph so result files cannot
        # imply that the future-solar reference contributed to prediction.
        model.reference_strength = 0.0
        model.reference_residual_active = False
    # Baselines may receive deterministic solar geometry solely so that the
    # same target-free night projection can be applied to every model.  The
    # tensor is deliberately withheld from their forecasting forward pass.
    model.projection_only_future = (
        projection_uses_future and not model_uses_future
    )
    model.evaluation_projection = args.evaluation_projection
    if ((name == "reference_ensemble_probglobal"
            or name == "pcarr_v3_frozen_ensemble" or name.startswith("pcarr_v4_")
            or name == "pcarr_v3_frozen_ensemble_edal"
            or name.startswith("pcarr_v5_") or name.startswith("pcarr_v6_")
            or name.startswith("pcarr_v7_") or name.startswith("pcarr_v8_")
            or name.startswith("pcarr_v9_") or name.startswith("pcarr_v10_")
            or name.startswith("pcarr_v11_") or name.startswith("pcarr_v12_")
            or name.startswith("pcarr_v13_") or name.startswith("pcarr_v14_")
            or name.startswith("pcarr_v15_") or name.startswith("pcarr_v16_")
            or name.startswith("pcarr_v17_") or name.startswith("pcarr_v18_")
            or name.startswith("pcarr_v19_") or name.startswith("pcarr_v20_")
            or name.startswith("pcarr_v21_") or name.startswith("pcarr_v23_"))
            and getattr(model, "reference_logits", None) is not None):
        calibrated_logits = calibrate_reference_ensemble(args, loaders, stats, device)
        with torch.no_grad():
            model.reference_logits.copy_(calibrated_logits)
        model.reference_logits.requires_grad_(False)
    reference_tune = (
        reference_residual_model
        and args.anchor_output_mode != "direct"
        and getattr(getattr(model, "anchor", None), "config", None) is not None
    )
    reference_base_test = None
    reference_base_val = None
    if reference_tune:
        # Stage 1 is the exact history-only forecaster.  The reference gate is
        # absent from the execution graph and its parameters are frozen.
        model.reference_residual_active = False
        model.reference_strength = 0.0
        for param_name, param in model.named_parameters():
            if param_name.startswith("reference_reliability_"):
                param.requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fsra_tune, _ = adapter_stage_policy(
        skip_adapter_stages=args.skip_adapter_stages,
        fsra_requested=fsra_model,
        prompt_requested=False,
    )
    fsra_base_test = None
    fsra_base_val = None
    prompt_base_test = None
    prompt_base_val = None
    fsra_loaded_metadata = None
    if fsra_model:
        # Stage 1 is an exactly matched numerical backbone.  Merely evaluating
        # a zero residual attention layer would consume dropout RNG and make
        # the control non-identical, so the adapter is skipped and frozen.
        model.fsra_enabled = False
        if args.skip_adapter_stages:
            model.fsra_strength = 0.0
        for param_name, param in model.named_parameters():
            if param_name.startswith("fsra_"):
                param.requires_grad_(False)
    checkpoint_average_used = 1
    checkpoint_average_selected = False
    pretraining = {
        "enabled": False,
        "epochs": 0,
        "epoch_losses": [],
        "span_lengths": [],
        "channel_indices": [],
    }
    backbone_history = []
    backbone_epochs_ran = 0
    backbone_best_epoch = 0
    backbone_early_stopped = False
    backbone_stop_reason = "not_trainable"
    auxiliary_training_history = []
    started = time.perf_counter()
    if trainable:
        prompt_requested = (
            name in {"pcarr_v6_prompt_tune", "pcarr_v7_fullprompt_tune"}
            or name.startswith("pcarr_v8_fullprompt_")
            or name.startswith("pcarr_v9_promptonly_")
            or name.startswith("pcarr_v10_meanlast_")
            or name.startswith("pcarr_v11_aligned_")
            or name.startswith("pcarr_v12_zerosafe_")
            or name.startswith("pcarr_v13_t3align")
            or cma_model
            or scale_prompt_model
        )
        _, prompt_tune = adapter_stage_policy(
            skip_adapter_stages=args.skip_adapter_stages,
            fsra_requested=False,
            prompt_requested=prompt_requested,
        )
        if prompt_requested and not prompt_tune:
            model.semantic_enabled = False
            model.semantic_strength = 0.0
            for param_name, param in model.named_parameters():
                if (param_name.startswith("semantic_")
                        or param_name.startswith("scale_prompt_")):
                    param.requires_grad_(False)
        joint_cma_training = (
            prompt_tune
            and cma_model
            and args.ours_cma_training == "joint"
        )
        if prompt_tune:
            model.semantic_enabled = joint_cma_training
            model.semantic_strength = 1.0 if joint_cma_training else 0.0
            if not joint_cma_training:
                for param_name, param in model.named_parameters():
                    if (param_name.startswith("semantic_")
                            or param_name.startswith("scale_prompt_")):
                        param.requires_grad_(False)
        pretraining = run_masked_history_pretraining(
            model, loaders[0], device, args,
        )
        actual_lr = float(getattr(model, "recommended_lr", args.lr))
        reference_logits = getattr(model, "reference_logits", None)
        semantic_schedule = name in {
            "pcarr_v5_prompt_sched", "pcarr_v5_prompt_align"
        }
        scheduler = None
        if semantic_schedule:
            semantic_params = [p for n, p in model.named_parameters()
                               if n.startswith("semantic_") and p.requires_grad]
            semantic_ids = {id(p) for p in semantic_params}
            base_params = [p for p in model.parameters()
                           if p.requires_grad and id(p) not in semantic_ids]
            opt = torch.optim.AdamW([
                {"params": base_params, "lr": 0.5 * actual_lr},
                {"params": semantic_params, "lr": 2.0 * actual_lr},
            ], weight_decay=args.weight_decay)
            total_steps = max(1, args.epochs * len(loaders[0]))
            warmup_steps = max(1, int(0.1 * total_steps))
            def lr_factor(step):
                if step < warmup_steps:
                    return float(step + 1) / warmup_steps
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                return 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
        elif reference_logits is not None:
            other_params = [p for p in model.parameters()
                            if p.requires_grad and p is not reference_logits]
            groups = []
            if other_params:
                groups.append({"params": other_params, "lr": actual_lr})
            groups.append({"params": [reference_logits], "lr": max(actual_lr, 5e-2)})
            optimizer_cls = (torch.optim.AdamW if args.optimizer == "adamw"
                             else torch.optim.Adam)
            opt = optimizer_cls(groups, weight_decay=args.weight_decay)
        else:
            optimizer_cls = (torch.optim.AdamW if args.optimizer == "adamw"
                             else torch.optim.Adam)
            if joint_cma_training:
                semantic_params = [
                    parameter for parameter_name, parameter in model.named_parameters()
                    if parameter.requires_grad
                    and (parameter_name.startswith("semantic_")
                         or parameter_name.startswith("scale_prompt_"))
                ]
                semantic_ids = {id(parameter) for parameter in semantic_params}
                numerical_params = [
                    parameter for parameter in model.parameters()
                    if parameter.requires_grad and id(parameter) not in semantic_ids
                ]
                opt = optimizer_cls(
                    [
                        {"params": numerical_params, "lr": actual_lr},
                        {
                            "params": semantic_params,
                            "lr": actual_lr * args.ours_cma_lr_multiplier,
                        },
                    ],
                    weight_decay=args.weight_decay,
                )
            else:
                opt = optimizer_cls(model.parameters(), lr=actual_lr,
                                    weight_decay=args.weight_decay)
        if scheduler is None and args.base_scheduler != "none":
            total_steps = max(1, args.epochs * len(loaders[0]))
            warmup_steps = (
                max(1, int(args.warmup_ratio * total_steps))
                if args.base_scheduler == "warmup_cosine" else 0
            )

            def base_lr_factor(step):
                if warmup_steps and step < warmup_steps:
                    return float(step + 1) / warmup_steps
                progress = ((step - warmup_steps)
                            / max(1, total_steps - warmup_steps))
                return 0.5 * (1.0 + np.cos(np.pi * min(max(progress, 0.0), 1.0)))

            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, base_lr_factor)
        best_score, best_state, best_epoch = float("inf"), None, -1
        top_base_states = []
        patience_left = args.patience
        for epoch in range(args.epochs):
            epoch_started = time.perf_counter()
            epoch_loss_sum = 0.0
            epoch_loss_batches = 0
            model.train()
            for batch in loaders[0]:
                xb, yb, ymb, cb, fxb, prompt_x = unpack_batch(batch, device)
                opt.zero_grad(set_to_none=True)
                pred = predict(model, xb, cb, fxb, prompt_x)
                loss = physical_loss(
                    pred, yb, loss_kind, stats, ymb, future_x=fxb,
                    daylight_weight=args.daylight_loss_weight,
                    ramp_weight=args.ramp_loss_weight,
                )
                if ("routeaux" in name
                        or (name in {
                            "ours",
                            "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma",
                        }
                            and args.route_loss_weight > 0.0)):
                    loss = loss + args.route_loss_weight * physical_routing_loss(model, xb, stats)
                if "stateaux" in name:
                    loss = loss + args.state_loss_weight * physical_state_auxiliary_loss(model, xb, stats)
                if ("sparse_router" in name
                        or (name in {
                            "ours",
                            "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma",
                        }
                            and args.router_specialization_weight > 0.0)):
                    loss = loss + args.router_specialization_weight * routing_specialization_loss(model)
                if args.candidate_aux or "candidate_aux" in name:
                    candidate = model.last_candidate_pred
                    candidate_mask = _training_point_mask(
                        yb, ymb, stats, args.daylight_loss_weight,
                    )
                    loss = loss + args.candidate_loss_weight * _masked_mean(
                        (candidate - yb).square(), candidate_mask,
                    )
                if name.endswith("_align") and model.semantic_alignment_loss is not None:
                    loss = loss + 0.01 * model.semantic_alignment_loss
                if (fsra_model
                        and model.fsra_alignment_loss is not None):
                    loss = loss + args.fsra_loss_weight * model.fsra_alignment_loss
                # A calibrated deterministic reference wrapped only by an
                # uncertainty head has no trainable mean-path tensor.  Its
                # point loss is intentionally constant until the dedicated
                # CRPS stage below.
                if not loss.requires_grad:
                    continue
                epoch_loss_sum += float(loss.detach().cpu().item())
                epoch_loss_batches += 1
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if scheduler is not None:
                    scheduler.step()
            val = evaluate(model, loaders[1], device, stats)
            val_score = _selection_score(val, args.selection_metric)
            improved = val_score < best_score - 1e-8
            current_state = None
            if args.checkpoint_average_k > 1:
                current_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                top_base_states.append((val_score, epoch + 1, current_state))
                top_base_states.sort(key=lambda item: item[0])
                del top_base_states[args.checkpoint_average_k:]
            should_stop = False
            if improved:
                best_score, best_epoch = val_score, epoch + 1
                best_state = (current_state if current_state is not None else {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                })
                patience_left = args.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    should_stop = True
            backbone_epochs_ran = epoch + 1
            epoch_record = {
                "stage": "backbone",
                "epoch": epoch + 1,
                "epoch_budget": int(args.epochs),
                "train_loss": (
                    epoch_loss_sum / epoch_loss_batches
                    if epoch_loss_batches else None
                ),
                "validation_score": float(val_score),
                "validation_mse_standardized": float(val["mse_standardized"]),
                "validation_mae_standardized": float(val["mae_standardized"]),
                "learning_rate": float(opt.param_groups[0]["lr"]),
                "improved": bool(improved),
                "best_epoch_so_far": int(best_epoch),
                "patience_remaining": int(patience_left),
                "seconds": float(time.perf_counter() - epoch_started),
            }
            backbone_history.append(epoch_record)
            print(
                "EPOCH " + json.dumps(epoch_record, ensure_ascii=False),
                flush=True,
            )
            if should_stop:
                backbone_early_stopped = True
                backbone_stop_reason = "validation_patience_exhausted"
                print(
                    "EARLY_STOP " + json.dumps(
                        {
                            "stage": "backbone",
                            "stopped_epoch": int(backbone_epochs_ran),
                            "best_epoch": int(best_epoch),
                            "patience": int(args.patience),
                            "selection_metric": args.selection_metric,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                break
        if not backbone_early_stopped:
            backbone_stop_reason = "epoch_budget_reached"
        backbone_best_epoch = int(best_epoch)
        print(
            "TRAINING_STOP " + json.dumps(
                {
                    "stage": "backbone",
                    "epochs_ran": int(backbone_epochs_ran),
                    "epoch_budget": int(args.epochs),
                    "best_epoch": int(backbone_best_epoch),
                    "early_stopped": bool(backbone_early_stopped),
                    "stop_reason": backbone_stop_reason,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if best_state is None:
            raise RuntimeError(
                "training produced no finite validation checkpoint; "
                f"last_validation={val!r}, last_selection_score={val_score!r}"
            )
        model.load_state_dict(best_state)
        if args.checkpoint_average_k > 1 and len(top_base_states) > 1:
            averaged_state = _average_state_dicts(
                [item[2] for item in top_base_states]
            )
            model.load_state_dict(averaged_state)
            averaged_val = evaluate(model, loaders[1], device, stats)
            averaged_score = _selection_score(
                averaged_val, args.selection_metric,
            )
            # Validation-only safety fallback: averaging is used only when it
            # improves the same metric that selected the individual checkpoint.
            if averaged_score < best_score - 1e-8:
                best_state = averaged_state
                checkpoint_average_used = len(top_base_states)
                checkpoint_average_selected = True
            else:
                model.load_state_dict(best_state)
        if reference_tune:
            # Fit only the history-conditioned reliability gate on top of the
            # frozen backbone. Validation strength zero is an exact fallback.
            model.reference_residual_active = False
            model.reference_strength = 0.0
            reference_base_val = evaluate(model, loaders[1], device, stats)
            reference_base_test = (
                None if args.validation_only
                else evaluate(model, loaders[2], device, stats)
            )
            for param in model.parameters():
                param.requires_grad_(False)
            reference_params = []
            for param_name, param in model.named_parameters():
                if param_name.startswith("reference_reliability_"):
                    param.requires_grad_(True)
                    reference_params.append(param)
            if not reference_params:
                raise RuntimeError(
                    "reference reliability tuning requested but no gate parameters exist"
                )
            model.reference_residual_active = True
            model.reference_strength = 1.0
            reference_opt = torch.optim.AdamW(
                reference_params, lr=actual_lr, weight_decay=args.weight_decay,
            )
            reference_epochs = max(3, min(6, args.epochs))
            reference_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                reference_opt, T_max=max(1, reference_epochs), eta_min=0.1 * actual_lr,
            )
            reference_best_score = float("inf")
            reference_best_state = None
            reference_best_epoch = -1
            for reference_epoch in range(reference_epochs):
                reference_epoch_started = time.perf_counter()
                reference_loss_sum = 0.0
                reference_loss_batches = 0
                model.train()
                for batch in loaders[0]:
                    xb, yb, ymb, cb, fxb, prompt_x = unpack_batch(batch, device)
                    reference_opt.zero_grad(set_to_none=True)
                    pred = predict(model, xb, cb, fxb, prompt_x)
                    point_mask = _training_point_mask(
                        yb, ymb, stats, args.daylight_loss_weight,
                    )
                    loss = physical_loss(
                        pred, yb, loss_kind, stats, ymb, future_x=fxb,
                        daylight_weight=args.daylight_loss_weight,
                        ramp_weight=args.ramp_loss_weight,
                    )
                    loss = loss + 0.10 * _masked_mean(
                        (pred - yb).abs(), point_mask,
                    )
                    reference_loss_sum += float(loss.detach().cpu().item())
                    reference_loss_batches += 1
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(reference_params, 1.0)
                    reference_opt.step()
                reference_scheduler.step()
                reference_val = evaluate(model, loaders[1], device, stats)
                reference_score = _selection_score(
                    reference_val, args.selection_metric,
                    reference=reference_base_val,
                )
                reference_improved = reference_score < reference_best_score
                if reference_improved:
                    reference_best_score = reference_score
                    reference_best_epoch = reference_epoch + 1
                    reference_best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                reference_record = {
                    "stage": "reference_reliability",
                    "epoch": reference_epoch + 1,
                    "epoch_budget": int(reference_epochs),
                    "train_loss": (
                        reference_loss_sum / reference_loss_batches
                        if reference_loss_batches else None
                    ),
                    "validation_score": float(reference_score),
                    "validation_mse_standardized": float(
                        reference_val["mse_standardized"]
                    ),
                    "validation_mae_standardized": float(
                        reference_val["mae_standardized"]
                    ),
                    "learning_rate": float(reference_opt.param_groups[0]["lr"]),
                    "improved": bool(reference_improved),
                    "best_epoch_so_far": int(reference_best_epoch),
                    "seconds": float(
                        time.perf_counter() - reference_epoch_started
                    ),
                }
                auxiliary_training_history.append(reference_record)
                print(
                    "STAGE_EPOCH "
                    + json.dumps(reference_record, ensure_ascii=False),
                    flush=True,
                )
            if reference_best_state is not None:
                model.load_state_dict(reference_best_state)
            best_reference_strength = 0.0
            best_reference_score = _selection_score(
                reference_base_val, args.selection_metric,
                reference=reference_base_val,
            )
            for strength in np.linspace(0.1, 1.5, 15):
                model.reference_strength = float(strength)
                strength_val = evaluate(model, loaders[1], device, stats)
                strength_score = _selection_score(
                    strength_val, args.selection_metric,
                    reference=reference_base_val,
                )
                if strength_score < best_reference_score:
                    best_reference_score = strength_score
                    best_reference_strength = float(strength)
            model.reference_strength = best_reference_strength
            model.reference_residual_active = best_reference_strength > 0.0
            if reference_best_epoch > 0:
                best_epoch += reference_best_epoch
        if fsra_tune:
            # Stage 2 freezes the verified history forecaster and fits only the
            # metadata adapter.  This makes strength zero the exact stage-1
            # checkpoint, rather than a backbone already altered by semantics.
            model.fsra_enabled = False
            base_val = evaluate(model, loaders[1], device, stats)
            fsra_base_val = base_val
            # Reporting-only paired baseline from the identical checkpoint.
            # It is never used for fitting, early stopping or strength choice.
            # The paired control is a diagnostic, so keep it on validation.
            # The held-out test set is evaluated exactly once for the headline result.
            fsra_base_test = None
            for param in model.parameters():
                param.requires_grad_(False)
            if args.fsra_load:
                fsra_loaded_metadata = _load_fsra_adapter(model, args.fsra_load).get(
                    "metadata", {},
                )
            fsra_params = []
            for param_name, param in model.named_parameters():
                if param_name.startswith("fsra_"):
                    param.requires_grad_(not args.fsra_eval_only)
                    fsra_params.append(param)
            if args.fsra_eval_only:
                if not args.fsra_load:
                    raise ValueError("--fsra-eval-only requires --fsra-load")
                model.fsra_enabled = model.fsra_has_content
                model.fsra_strength = float(args.fsra_eval_strength)
            elif model.fsra_has_content and fsra_params:
                model.fsra_enabled = True
                model.fsra_strength = 1.0
                tune_opt = torch.optim.AdamW(
                    fsra_params, lr=args.lr, weight_decay=args.weight_decay,
                )
                adaptive_fsra = getattr(model, "fsra_cma_heads", 0) > 0
                tune_epochs = (max(6, min(10, 2 * args.epochs)) if adaptive_fsra
                               else max(3, min(5, args.epochs)))
                tune_scheduler = (
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        tune_opt, T_max=max(1, tune_epochs), eta_min=0.1 * args.lr,
                    ) if adaptive_fsra else None
                )
                tune_best_score, tune_best_state, tune_best_epoch = float("inf"), None, -1
                for tune_epoch in range(tune_epochs):
                    fsra_epoch_started = time.perf_counter()
                    fsra_loss_sum = 0.0
                    fsra_loss_batches = 0
                    model.train()
                    for batch in loaders[0]:
                        xb, yb, ymb, cb, fxb, prompt_x = unpack_batch(batch, device)
                        tune_opt.zero_grad(set_to_none=True)
                        pred = predict(model, xb, cb, fxb, prompt_x)
                        loss = physical_loss(
                            pred, yb, loss_kind, stats, ymb, future_x=fxb,
                            daylight_weight=args.daylight_loss_weight,
                            ramp_weight=args.ramp_loss_weight,
                        )
                        if adaptive_fsra:
                            # The physical objective is MSE-dominant.  A small
                            # masked MAE term prevents the semantic adapter from
                            # buying lower squared error with a broad median
                            # bias, while leaving the frozen backbone untouched.
                            fsra_point_mask = _training_point_mask(
                                yb, ymb, stats, args.daylight_loss_weight,
                            )
                            loss = loss + args.fsra_mae_weight * _masked_mean(
                                (pred - yb).abs(), fsra_point_mask,
                            )
                        if model.fsra_alignment_loss is not None:
                            loss = loss + args.fsra_loss_weight * model.fsra_alignment_loss
                        fsra_loss_sum += float(loss.detach().cpu().item())
                        fsra_loss_batches += 1
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(fsra_params, 1.0)
                        tune_opt.step()
                    if tune_scheduler is not None:
                        tune_scheduler.step()
                    tune_val = evaluate(model, loaders[1], device, stats)
                    tune_score = _selection_score(
                        tune_val, args.selection_metric,
                        reference=base_val if adaptive_fsra else None,
                    )
                    fsra_improved = tune_score < tune_best_score
                    if fsra_improved:
                        tune_best_score = tune_score
                        tune_best_epoch = tune_epoch + 1
                        tune_best_state = {
                            key: value.detach().cpu().clone()
                            for key, value in model.state_dict().items()
                        }
                    fsra_record = {
                        "stage": "fsra_adapter",
                        "epoch": tune_epoch + 1,
                        "epoch_budget": int(tune_epochs),
                        "train_loss": (
                            fsra_loss_sum / fsra_loss_batches
                            if fsra_loss_batches else None
                        ),
                        "validation_score": float(tune_score),
                        "validation_mse_standardized": float(
                            tune_val["mse_standardized"]
                        ),
                        "validation_mae_standardized": float(
                            tune_val["mae_standardized"]
                        ),
                        "learning_rate": float(tune_opt.param_groups[0]["lr"]),
                        "improved": bool(fsra_improved),
                        "best_epoch_so_far": int(tune_best_epoch),
                        "seconds": float(time.perf_counter() - fsra_epoch_started),
                    }
                    auxiliary_training_history.append(fsra_record)
                    print(
                        "STAGE_EPOCH "
                        + json.dumps(fsra_record, ensure_ascii=False),
                        flush=True,
                    )
                if tune_best_state is not None:
                    model.load_state_dict(tune_best_state)
                    best_epoch = args.epochs + tune_best_epoch
                # Validation-only safety calibration covers both the residual
                # and relation bias.  Test targets are never queried.
                best_strength, best_strength_score = 0.0, _selection_score(
                    base_val, args.selection_metric,
                    reference=base_val if adaptive_fsra else None,
                )
                for strength in np.linspace(0.1, 1.5, 15):
                    model.fsra_strength = float(strength)
                    strength_val = evaluate(model, loaders[1], device, stats)
                    strength_score = _selection_score(
                        strength_val, args.selection_metric,
                        reference=base_val if adaptive_fsra else None,
                    )
                    if strength_score < best_strength_score:
                        best_strength_score = strength_score
                        best_strength = float(strength)
                model.fsra_strength = best_strength
                model.fsra_enabled = best_strength > 0.0
            else:
                model.fsra_strength = 0.0
                model.fsra_enabled = False
            if args.fsra_save:
                _save_fsra_adapter(model, args.fsra_save, {
                    "source_data": str(args.data),
                    "model": name,
                    "seed": int(args.seed),
                    "seq_len": int(args.seq_len),
                    "fsra_strength_selected_on_source_validation": float(model.fsra_strength),
                    "loaded_from": str(args.fsra_load) if args.fsra_load else None,
                    "target_semantic_training": not bool(args.fsra_eval_only),
                })
        if prompt_tune:
            # Low-risk stage 2: keep the verified history/physics forecaster
            # fixed and fit only the frozen-GPT-2 adapter and its residual gate.
            delayed_prompt_attach = (
                (name.startswith("pcarr_v16_fsra")
                 or name.startswith("pcarr_v17_fsra")
                 or name.startswith("pcarr_v18_fsra")
                 or name.startswith("pcarr_v19_fsra")
                 or name.startswith("pcarr_v21_fsra")
                 or name.startswith("pcarr_v22_fsra"))
                and model.scale_prompt_adapter is None
                and model.semantic_hierarchical_router is None
            )
            model.semantic_enabled = False if delayed_prompt_attach else True
            model.semantic_strength = 0.0
            base_val = evaluate(model, loaders[1], device, stats)
            prompt_base_val = base_val
            # Validation already provides the fixed-checkpoint paired control.
            # Avoid repeated diagnostic passes over the held-out test distribution.
            prompt_base_test = None
            if delayed_prompt_attach:
                if (name.startswith("pcarr_v21_fsra")
                        or name.startswith("pcarr_v22_fsra")):
                    model.attach_hierarchical_prompt_router(
                        prompt_dim=(8 if prompt_stats else 1536),
                        centered_route_delta=name.startswith("pcarr_v22_fsra"),
                        content_pairing=name.endswith("_c"),
                    )
                elif name.startswith("pcarr_v19_fsra"):
                    model.attach_tf_prompt_router(
                        prompt_dim=(8 if prompt_stats else 1792),
                    )
                else:
                    model.attach_scale_prompt_adapter(
                        prompt_dim=(8 if prompt_stats else
                                    1792 if name.startswith("pcarr_v18_fsra") else 1536),
                        n_heads=3,
                        output_gate=True,
                        contrastive_scale_alignment=name.startswith("pcarr_v17_fsra"),
                    )
                model.semantic_enabled = True
            model.semantic_strength = 1.0
            for param in model.parameters():
                param.requires_grad_(False)
            semantic_params = []
            for param_name, param in model.named_parameters():
                if (param_name.startswith("semantic_")
                        or param_name.startswith("scale_prompt_")):
                    param.requires_grad_(True)
                    semantic_params.append(param)
            if not semantic_params:
                raise RuntimeError("prompt tuning requested but no prompt-adapter parameters exist")
            if name.startswith("pcarr_v22_fsra"):
                slow_params, fast_params = [], []
                for param_name, param in model.named_parameters():
                    if not param.requires_grad:
                        continue
                    if (".tf_delta." in param_name
                            or ".scale_delta." in param_name):
                        fast_params.append(param)
                    else:
                        slow_params.append(param)
                tune_opt = torch.optim.AdamW(
                    [
                        {"params": slow_params, "lr": 0.2 * actual_lr},
                        {"params": fast_params, "lr": actual_lr},
                    ],
                    weight_decay=args.weight_decay,
                )
                tune_epochs = max(8, min(10, args.epochs))
            else:
                prompt_lr = (
                    actual_lr * args.ours_cma_lr_multiplier
                    if cma_model else actual_lr
                )
                tune_opt = torch.optim.AdamW(
                    semantic_params, lr=prompt_lr, weight_decay=args.weight_decay,
                )
                tune_epochs = (
                    int(args.ours_cma_epochs)
                    if cma_model and args.ours_cma_epochs > 0
                    else max(3, min(6, args.epochs))
                )
            tune_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                tune_opt,
                T_max=max(1, tune_epochs * len(semantic_train_loader)),
                eta_min=0.1 * tune_opt.param_groups[0]["lr"],
            )
            tune_best_score, tune_best_state, tune_best_epoch = float("inf"), None, -1
            for tune_epoch in range(tune_epochs):
                prompt_epoch_started = time.perf_counter()
                prompt_loss_sum = 0.0
                prompt_loss_batches = 0
                model.train()
                for batch in semantic_train_loader:
                    xb, yb, ymb, cb, fxb, prompt_x = unpack_batch(batch, device)
                    tune_opt.zero_grad(set_to_none=True)
                    pred = predict(model, xb, cb, fxb, prompt_x)
                    loss = physical_loss(
                        pred, yb, loss_kind, stats, ymb, future_x=fxb,
                        daylight_weight=args.daylight_loss_weight,
                        ramp_weight=args.ramp_loss_weight,
                    )
                    semantic_point_mask = _training_point_mask(
                        yb, ymb, stats, args.daylight_loss_weight,
                    )
                    if (name.startswith("pcarr_v8_fullprompt_")
                            or name.startswith("pcarr_v9_promptonly_")):
                        loss = loss + args.semantic_mae_weight * _masked_mean(
                            (pred - yb).abs(), semantic_point_mask,
                        )
                    elif (name.startswith("pcarr_v10_meanlast_")
                          or name.startswith("pcarr_v11_aligned_")
                          or name.startswith("pcarr_v12_zerosafe_")
                          or cma_model
                          or scale_prompt_model):
                        loss = loss + args.semantic_mae_weight * _masked_mean(
                            (pred - yb).abs(), semantic_point_mask,
                        )
                    if ((name.startswith("pcarr_v11_aligned_")
                         or name.startswith("pcarr_v12_zerosafe_")
                         or scale_prompt_model)
                            and model.semantic_alignment_loss is not None):
                        alignment_weight = (
                            0.02 if name.endswith("_c") else
                            0.50 if name.startswith("pcarr_v12_zerosafe_") else
                            0.05 if name.startswith("pcarr_v17_fsra") else
                            0.01 if scale_prompt_model else 0.05
                        )
                        loss = loss + alignment_weight * model.semantic_alignment_loss
                    prompt_loss_sum += float(loss.detach().cpu().item())
                    prompt_loss_batches += 1
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(semantic_params, 1.0)
                    tune_opt.step(); tune_scheduler.step()
                tune_val = evaluate(model, loaders[1], device, stats)
                tune_score = _selection_score(
                    tune_val, args.selection_metric, reference=base_val,
                )
                prompt_improved = tune_score < tune_best_score
                if prompt_improved:
                    tune_best_score = tune_score
                    tune_best_epoch = tune_epoch + 1
                    tune_best_state = {
                        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                    }
                prompt_record = {
                    "stage": "prompt_adapter",
                    "epoch": tune_epoch + 1,
                    "epoch_budget": int(tune_epochs),
                    "train_loss": (
                        prompt_loss_sum / prompt_loss_batches
                        if prompt_loss_batches else None
                    ),
                    "validation_score": float(tune_score),
                    "validation_mse_standardized": float(
                        tune_val["mse_standardized"]
                    ),
                    "validation_mae_standardized": float(
                        tune_val["mae_standardized"]
                    ),
                    "learning_rate": float(tune_opt.param_groups[0]["lr"]),
                    "improved": bool(prompt_improved),
                    "best_epoch_so_far": int(tune_best_epoch),
                    "seconds": float(time.perf_counter() - prompt_epoch_started),
                }
                auxiliary_training_history.append(prompt_record)
                print(
                    "STAGE_EPOCH "
                    + json.dumps(prompt_record, ensure_ascii=False),
                    flush=True,
                )
            if tune_best_state is None:
                raise RuntimeError("prompt-only fine-tuning did not produce a checkpoint")
            model.load_state_dict(tune_best_state)
            # Leakage-free residual calibration.  The exact backbone is gamma
            # zero; the test split is never queried during this search.
            best_strength, best_strength_score = 0.0, float("inf")
            strength_grid = (
                np.linspace(0.0, 1.0, 11)
                if name.startswith("pcarr_v22_fsra")
                else np.linspace(
                    0.0,
                    args.ours_semantic_strength_max,
                    args.ours_semantic_strength_steps,
                ) if cma_model
                else np.linspace(0.0, 1.5, 16)
            )
            for strength in strength_grid:
                model.semantic_strength = float(strength)
                strength_val = evaluate(model, loaders[1], device, stats)
                strength_score = _selection_score(
                    strength_val, args.selection_metric, reference=base_val,
                )
                if strength_score < best_strength_score:
                    best_strength_score = strength_score
                    best_strength = float(strength)
            model.semantic_strength = best_strength
            best_epoch = args.epochs + tune_best_epoch
        if getattr(model, "probabilistic_output", False):
            # GenSolar-style two-stage separation in a form appropriate for a
            # single-site forecaster: freeze the deterministic mean (including
            # FSRA) and fit only a Gaussian residual distribution.  This keeps
            # point predictions exactly unchanged and prevents uncertainty
            # calibration from buying coverage by degrading the mean.
            for param in model.parameters():
                param.requires_grad_(False)
            uncertainty_params = []
            for param_name, param in model.named_parameters():
                if param_name.startswith("uncertainty_"):
                    param.requires_grad_(True)
                    uncertainty_params.append(param)
            uncertainty_opt = torch.optim.AdamW(
                uncertainty_params, lr=args.lr, weight_decay=1e-4,
            )
            uncertainty_epochs = max(3, min(6, args.epochs))
            initial_prob_val = evaluate(model, loaders[1], device, stats)
            uncertainty_best_crps = initial_prob_val["gaussian_crps_standardized"]
            uncertainty_best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            uncertainty_best_epoch = 0
            for uncertainty_epoch in range(uncertainty_epochs):
                model.eval()
                if model.uncertainty_head is not None:
                    model.uncertainty_head.train()
                for batch in loaders[0]:
                    xb, yb, ymb, cb, fxb, prompt_x = unpack_batch(batch, device)
                    uncertainty_opt.zero_grad(set_to_none=True)
                    pred = predict(model, xb, cb, fxb, prompt_x).detach()
                    scale = model.last_predictive_scale
                    crps = _gaussian_crps_torch(pred, yb, scale)
                    loss = _masked_mean(crps, ymb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(uncertainty_params, 1.0)
                    uncertainty_opt.step()
                uncertainty_val = evaluate(model, loaders[1], device, stats)
                uncertainty_crps = uncertainty_val["gaussian_crps_standardized"]
                if uncertainty_crps < uncertainty_best_crps:
                    uncertainty_best_crps = uncertainty_crps
                    uncertainty_best_epoch = uncertainty_epoch + 1
                    uncertainty_best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
            model.load_state_dict(uncertainty_best_state)
            calibration = _validation_conformal_scale(
                model, loaders[1], device,
            )
            model.uncertainty_calibration.fill_(calibration)
            best_epoch = best_epoch + uncertainty_best_epoch
        if getattr(model, "quantile_output", False):
            # Quantile regression is a separate probabilistic baseline.  As
            # with the Gaussian adapter, the deterministic mean and FSRA are
            # frozen so interval quality cannot be bought by changing MAE/MSE.
            for param in model.parameters():
                param.requires_grad_(False)
            quantile_params = []
            for param_name, param in model.named_parameters():
                if param_name.startswith("quantile_"):
                    param.requires_grad_(True)
                    quantile_params.append(param)
            quantile_opt = torch.optim.AdamW(
                quantile_params, lr=args.lr, weight_decay=1e-4,
            )
            quantile_epochs = max(3, min(6, args.epochs))
            initial_quantile_val = evaluate(model, loaders[1], device, stats)
            quantile_best_crps = initial_quantile_val["quantile_crps_standardized"]
            quantile_best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            quantile_best_epoch = 0
            levels = model.quantile_levels.view(1, 1, -1)
            for quantile_epoch in range(quantile_epochs):
                model.eval()
                if model.quantile_head is not None:
                    model.quantile_head.train()
                for batch in loaders[0]:
                    xb, yb, ymb, cb, fxb, prompt_x = unpack_batch(batch, device)
                    quantile_opt.zero_grad(set_to_none=True)
                    pred = predict(model, xb, cb, fxb, prompt_x).detach()
                    members = model.last_predictive_members
                    error = yb.unsqueeze(-1) - members
                    pinball = torch.maximum(levels * error, (levels - 1.0) * error)
                    mask3 = ymb.unsqueeze(-1).expand_as(pinball)
                    loss = _masked_mean(pinball, mask3)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(quantile_params, 1.0)
                    quantile_opt.step()
                quantile_val = evaluate(model, loaders[1], device, stats)
                quantile_crps = quantile_val["quantile_crps_standardized"]
                if quantile_crps < quantile_best_crps:
                    quantile_best_crps = quantile_crps
                    quantile_best_epoch = quantile_epoch + 1
                    quantile_best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
            model.load_state_dict(quantile_best_state)
            adjustment = _validation_quantile_adjustment(
                model, loaders[1], device,
            )
            model.quantile_interval_adjustment.fill_(adjustment)
            best_epoch = best_epoch + quantile_best_epoch
        if (args.calibrate_output_scale
                and bool(getattr(model, "output_anchor", False))):
            calibration_metric = (
                args.selection_metric if args.calibration_metric == "inherit"
                else args.calibration_metric
            )
            best_output_scale = 1.0
            best_output_score = float("inf")
            for output_scale in np.linspace(0.5, 3.0, 11):
                model.output_correction_scale = float(output_scale)
                scale_val = evaluate(model, loaders[1], device, stats)
                scale_score = _selection_score(scale_val, calibration_metric)
                if scale_score < best_output_score:
                    best_output_score = scale_score
                    best_output_scale = float(output_scale)
            model.output_correction_scale = best_output_scale
    else:
        best_epoch = 0
        actual_lr = 0.0
    val = evaluate(model, loaders[1], device, stats)
    test = (None if args.validation_only else evaluate(
        model, loaders[2], device, stats, analyze_regimes=True,
        include_preview=True,
    ))
    prompt_counterfactuals = None
    if ((name.startswith("pcarr_v22_fsra") or cma_model)
            and prompt_by_start is not None):
        # Fixed-weight semantic falsification: train only with correct prompts,
        # lock every parameter and the validation-selected strength, then alter
        # only the prompt bank at inference.  This cannot reward an arbitrary
        # codebook through separately retrained control models.
        eval_indices = va if args.validation_only else te
        prompt_counterfactuals = {}
        control_banks = {
            "shuffle": prompt_by_start,
            "random": _matched_random_prompt_bank(prompt_by_start),
            "zero": {
                start: np.zeros_like(value)
                for start, value in prompt_by_start.items()
            },
        }
        for control, bank in control_banks.items():
            dataset = WindowSet(
                x, eval_indices, args.seq_len, args.horizon, args.cycle_len,
                cycle_ids, stats.get("target_observed"),
                include_future=include_future, future_indices=audited_future,
                prompt_by_start=bank, prompt_shuffle=(control == "shuffle"),
            )
            control_loader = DataLoader(
                dataset, args.batch, shuffle=False, num_workers=0,
            )
            control_result = evaluate(
                model, control_loader, device, stats,
                analyze_regimes=not args.validation_only,
            )
            prompt_counterfactuals[control] = {
                key: control_result[key] for key in (
                    "mse_standardized", "mae_standardized", "rmse_physical",
                    "mae_physical", "nmae_capacity", "ramp_mae_norm",
                )
            }
    if "mcdropout" in name:
        mc_calibration = _validation_mc_dropout_scale(
            model, loaders[1], device, seed=args.seed,
        )
        val.update(evaluate_mc_dropout(
            model, loaders[1], device, stats,
            calibration=mc_calibration, seed=args.seed,
        ))
        if test is not None:
            test.update(evaluate_mc_dropout(
                model, loaders[2], device, stats,
                calibration=mc_calibration, seed=args.seed + 1000,
            ))
    fsra_counterfactuals = None
    if (name in {"pcarr_v14_fsra", "pcarr_v14_fsra_roles",
                  "pcarr_v14_fsra_residualonly",
                 "pcarr_v14_fsra_roles_residualonly",
                 "pcarr_v14_fsra_roles_adaptive3_residualonly",
                 "pcarr_v14_fsra_roles_adaptive3_paired_residualonly",
                 "pcarr_v14_fsra_roles_adaptive3_paired_probglobal_residualonly",
                 "pcarr_v14_fsra_roles_adaptive3_paired_probstate_residualonly",
                 "pcarr_v14_fsra_roles_adaptive3_paired_probquantile_residualonly",
                 "pcarr_v14_fsra_roles_adaptive3_paired_mcdropout_residualonly",
                 "pcarr_v14_fsra_roles_adaptive3_paired_decomp_residualonly",
                 "pcarr_v14_fsra_roles_adaptive3_paired_decomp_probglobal_residualonly",
                 "pcarr_v14_fsra_roles_adaptive3_paired_decomp_probstate_residualonly",
                 "pcarr_v14_fsra_roles_adaptive3_transfer_residualonly",
                 "pcarr_v14_fsra_roles_adaptive3_transfergrid_residualonly",
                  "pcarr_v14_fsra_roles_adaptive3_transfergrid_paired_residualonly",
                  "pcarr_v21_fsra_roles_adaptive3_paired_residualonly_refrel",
                  "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma",
                  "pcarr_v23_fsra_roles_adaptive3_paired_probstate_residualonly_refrel_t3cma",
                  "pcarr_v23_fsra_roles_adaptive3_paired_probquantile_residualonly_refrel_t3cma",
                  "pcarr_v23_fsra_roles_adaptive3_paired_mcdropout_residualonly_refrel_t3cma"}
            and model.fsra_channel_embeddings is not None
            and model.fsra_strength > 0.0):
        # Hold every trained weight fixed and corrupt only the frozen channel
        # metadata at inference.  This asks whether the fitted model actually
        # uses the correct semantic pairing, rather than whether a separately
        # trained network can adapt around an arbitrary codebook.
        original_embeddings = model.fsra_channel_embeddings.detach().clone()
        fsra_counterfactuals = {}
        for control in ("shuffle", "zero", "random"):
            replacement = controlled_channel_semantics(
                original_embeddings.cpu().numpy(), control, args.seed,
            )
            with torch.no_grad():
                model.fsra_channel_embeddings.copy_(
                    torch.tensor(replacement, device=device, dtype=original_embeddings.dtype)
                )
            model.fsra_has_content = bool(np.abs(replacement).sum() > 0.0)
            counter_loader = loaders[1]
            counter = evaluate(model, counter_loader, device, stats)
            fsra_counterfactuals[control] = {
                key: counter[key] for key in (
                    "mse_standardized", "mae_standardized", "rmse_physical",
                    "mae_physical", "nmae_capacity", "ramp_mae_norm",
                )
            }
        with torch.no_grad():
            model.fsra_channel_embeddings.copy_(original_embeddings)
        model.fsra_has_content = True
    counterfactuals = None
    if not args.validation_only and name in {
        "pcarr", "pcarr_candidate_aux",
        "pcarr_v14_fsra_roles_adaptive3_paired_residualonly",
        "pcarr_v14_fsra_roles_adaptive3_paired_probstate_residualonly",
        "pcarr_v23_fsra_roles_adaptive3_paired_residualonly_refrel_t3cma",
        "pcarr_v23_fsra_roles_adaptive3_paired_probstate_residualonly_refrel_t3cma",
    }:
        counterfactuals = {}
        diagnostic_loader = loaders[1]
        route_names = ["force_change", "force_spectral", "force_identity"]
        for route_index, route_name in enumerate(route_names):
            model.force_tf_route = route_index
            counterfactuals[route_name] = _compact_counterfactual(
                evaluate(model, diagnostic_loader, device, stats)
            )
        model.force_tf_route = None
        for band_index, band_name in enumerate(("force_slow", "force_medium", "force_fast")):
            model.spectral_mkan.force_band_index = band_index
            counterfactuals[band_name] = _compact_counterfactual(
                evaluate(model, diagnostic_loader, device, stats)
            )
        model.spectral_mkan.force_band_index = None
        for intervention in ("zero", "shuffle"):
            model.spectral_mkan.band_state_intervention = intervention
            counterfactuals[f"band_state_{intervention}"] = _compact_counterfactual(
                evaluate(model, diagnostic_loader, device, stats)
            )
        model.spectral_mkan.band_state_intervention = None
        for intervention in ("zero", "shuffle"):
            model.routing_state_intervention = intervention
            counterfactuals[f"full_state_{intervention}"] = _compact_counterfactual(
                evaluate(model, diagnostic_loader, device, stats)
            )
        model.routing_state_intervention = None
        for label, attribute in (
            ("tf", "tf_state_intervention"),
            ("scale", "scale_state_intervention"),
            ("output", "output_state_intervention"),
        ):
            setattr(model, attribute, "shuffle")
            counterfactuals[f"{label}_state_shuffle"] = _compact_counterfactual(
                evaluate(model, diagnostic_loader, device, stats)
            )
            setattr(model, attribute, None)
    patch_scales = getattr(getattr(model, "patch", None), "scales", None)
    if patch_scales is not None:
        patch_scales = [[int(p), int(s)] for p, s in patch_scales]
    spectral_indices = getattr(model, "frequency_indices", None)
    if spectral_indices is not None:
        spectral_indices = [int(i) for i in spectral_indices.detach().cpu().tolist()]
    spectral_features = ([stats["feature_names"][i] for i in spectral_indices]
                          if spectral_indices is not None else None)
    scale_token_module = getattr(model, "scale_token_refiner", None)
    scale_token_branch_weights = None
    scale_token_residual_scale = None
    if (scale_token_module is not None
            and scale_token_module.last_branch_weights is not None):
        scale_token_branch_weights = (
            scale_token_module.last_branch_weights.detach().cpu().tolist()
        )
        scale_token_residual_scale = torch.tanh(
            scale_token_module.residual_scale.detach()
        ).cpu().tolist()
    result = {"model": name, "seq_len": args.seq_len, "horizon": args.horizon,
            "sample_hours": stats["dt_hours"], "forecast_horizon_hours": stats["horizon_hours"],
            "seed": args.seed, "params": trainable, "best_epoch": best_epoch,
            "backbone_best_epoch": backbone_best_epoch,
            "backbone_epochs_ran": backbone_epochs_ran,
            "backbone_early_stopped": backbone_early_stopped,
            "backbone_stop_reason": backbone_stop_reason,
            "early_stopping_patience": int(args.patience),
            "backbone_training_history": backbone_history,
            "auxiliary_training_history": auxiliary_training_history,
            "frequency_v2_mode": getattr(model, "frequency_v2_mode", None),
            "frequency_v3_mode": getattr(model, "frequency_v3_mode", None),
            "frequency_v4_mode": getattr(model, "frequency_v4_mode", None),
            "frequency_v5_mode": getattr(model, "frequency_v5_mode", None),
            "frequency_v6_mode": getattr(model, "frequency_v6_mode", None),
            "frequency_v7_mode": getattr(model, "frequency_v7_mode", None),
            "frequency_v8_mode": getattr(model, "frequency_v8_mode", None),
            "frequency_v9_mode": getattr(model, "frequency_v9_mode", None),
            "spectral_mkan_mode": getattr(model, "spectral_mkan_mode", "legacy"),
            "clean_frequency_ablation": bool(
                getattr(model, "clean_frequency_ablation", False)
            ),
            "learning_rate": actual_lr,
            "backbone_epochs_requested": args.epochs,
            "checkpoint_average_k": int(args.checkpoint_average_k),
            "checkpoint_average_used": int(checkpoint_average_used),
            "checkpoint_average_selected": bool(checkpoint_average_selected),
            "candidate_auxiliary": bool(args.candidate_aux or "candidate_aux" in name),
            "candidate_loss_weight": (float(args.candidate_loss_weight)
                                      if args.candidate_aux or "candidate_aux" in name else 0.0),
            "output_correction_floor": float(args.output_correction_floor),
            "output_router_bias": args.output_router_bias,
            "anchor_output_mode": args.anchor_output_mode,
            "output_correction_scale": float(
                getattr(model, "output_correction_scale", 1.0)
            ),
            "ours_revin_mode": str(
                getattr(model, "revin_mode", "not_applicable")
            ),
            "local_token_refiner": bool(args.local_token_refiner),
            "local_refiner_dropout": float(args.local_refiner_dropout),
            "scale_token_mixer": bool(args.scale_token_mixer),
            "scale_token_mixer_adaptive": bool(
                args.scale_token_mixer and not args.fixed_scale_token_mixer
            ),
            "scale_token_mixer_dropout": float(args.scale_token_mixer_dropout),
            "scale_token_branch_weights": scale_token_branch_weights,
            "scale_token_residual_scale": scale_token_residual_scale,
            "fsra_mae_weight": float(args.fsra_mae_weight),
            "semantic_mae_weight": float(args.semantic_mae_weight),
            "selection_metric": args.selection_metric,
            "calibration_metric": args.calibration_metric,
            "daylight_loss_weight": float(args.daylight_loss_weight),
            "ramp_loss_weight": float(args.ramp_loss_weight),
            "router_activation": args.router_activation,
            "expert_drop_path": float(args.expert_drop_path),
            "gtr_dropout": float(args.gtr_dropout),
            "masked_history_pretraining": pretraining,
            "optimizer": args.optimizer,
            "batch_size": int(args.batch),
            "weight_decay": float(args.weight_decay),
            "base_scheduler": args.base_scheduler,
            "warmup_ratio": float(args.warmup_ratio),
            "validation_only": bool(args.validation_only),
            "skip_adapter_stages": bool(args.skip_adapter_stages),
            "evaluation_projection": str(args.evaluation_projection),
            "selected_train_windows": int(stats.get("selected_train_windows", len(tr))),
            "available_train_windows": int(
                stats.get("available_train_windows", len(tr))
            ),
            "train_samples_per_epoch": int(len(loaders[0].sampler)),
            "selected_val_windows": int(stats.get("selected_val_windows", len(va))),
            "selected_test_windows": int(stats.get("selected_test_windows", len(te))),
            "available_val_windows": int(
                stats.get("available_val_windows", len(va))
            ),
            "available_test_windows": int(
                stats.get("available_test_windows", len(te))
            ),
            "max_train_windows": int(stats.get("max_train_windows", args.max_windows)),
            "max_eval_windows": int(stats.get("max_eval_windows", args.max_windows)),
            "max_val_windows": int(stats.get("max_val_windows", -1)),
            "max_test_windows": int(stats.get("max_test_windows", -1)),
            "split_train_fraction": float(stats.get("split_train_fraction", 0.70)),
            "split_validation_end_fraction": float(
                stats.get("split_validation_end_fraction", 0.85)
            ),
            "train_sampling_policy": stats.get("train_sampling_policy", "uniform"),
            "ramp_enrichment_fraction": float(
                stats.get("ramp_enrichment_fraction", 0.0)
            ),
            "available_train_history_ramp_mean": float(
                stats.get("available_train_history_ramp_mean", 0.0)
            ),
            "selected_train_history_ramp_mean": float(
                stats.get("selected_train_history_ramp_mean", 0.0)
            ),
            "loss_kind": loss_kind,
            "seconds": time.perf_counter() - started, "val": val, "test": test,
            "capacity": stats["capacity"], "capacity_source": stats["capacity_source"],
            "observed_policy": stats["observed_policy"],
            "features": stats.get("feature_names"),
            "physics_features": stats.get("physics_features"),
            "quality_features": stats.get("quality_features"),
            "uses_future_covariates": bool(
                model_uses_future or prompt_uses_future_solar
            ),
            "tensor_path_uses_future_covariates": bool(model_uses_future),
            "prompt_uses_future_solar_geometry": bool(
                prompt_uses_future_solar
            ),
            "prompt_future_solar_policy": str(
                args.prompt_future_solar_policy
            ),
            "projection_uses_future_solar_geometry": bool(
                projection_uses_future
            ),
            "reference_path_uses_future_solar_geometry": bool(
                solar_future_model
            ),
            "future_feature_policy": (
                "deterministic_solar_geometry"
                if solar_future_model or prompt_uses_future_solar
                else stats.get("future_feature_policy")
                if model_uses_future else "none"
            ),
            "future_features": (
                stats.get("solar_future_features")
                if solar_future_model or prompt_uses_future_solar
                else stats.get("future_features")
                if model_uses_future and not name.endswith("_future") else []
            ),
            "patch_scales_steps": patch_scales,
            "patch_scales_hours": ([[p * stats["dt_hours"], s * stats["dt_hours"]]
                                     for p, s in patch_scales]
                                    if patch_scales is not None else None),
            "model_variant": getattr(model, "variant", None),
            "ours_hyperparameters": ({
                "d_model": int(args.d_model),
                "patch_hours": [float(value) for value in args.ours_patch_hours],
                "gtr_period": int(args.ours_gtr_period),
                "dropout": float(args.ours_dropout),
                "corpatch_heads": int(args.ours_corpatch_heads),
                "cma_heads": int(args.ours_cma_heads),
                "cma_dropout": float(args.ours_cma_dropout),
                "cma_training": str(args.ours_cma_training),
                "cma_prompt_features": str(args.ours_cma_prompt_features),
                "prompt_alignment": (
                    "linear_projection_norm"
                    if name == "ours"
                    else str(getattr(model, "cma_prompt_encoder", "transformer"))
                ),
                "cma_epochs": int(args.ours_cma_epochs),
                "cma_lr_multiplier": float(args.ours_cma_lr_multiplier),
                "semantic_strength_max": float(args.ours_semantic_strength_max),
                "semantic_strength_steps": int(args.ours_semantic_strength_steps),
                "learning_rate": float(actual_lr) if trainable else None,
                "route_loss_weight": float(args.route_loss_weight),
                "router_specialization_weight": float(
                    args.router_specialization_weight
                ),
            } if cma_model
               else None),
            "spectral_features": spectral_features,
            "counterfactuals": counterfactuals,
            "counterfactual_split": (
                "validation" if counterfactuals is not None else None
            ),
            "headline_test_evaluation_passes": (
                0 if args.validation_only else 1
            ),
            "fsra_counterfactuals": fsra_counterfactuals,
            "prompt_counterfactuals": prompt_counterfactuals,
            "prompt_preprocessing": prompt_preprocessing_metadata,
            "fsra_paired_backbone_test": fsra_base_test,
            "fsra_paired_backbone_val": fsra_base_val,
            "prompt_paired_backbone_test": prompt_base_test,
            "prompt_paired_backbone_val": prompt_base_val,
            "reference_paired_backbone_test": reference_base_test,
            "reference_paired_backbone_val": reference_base_val,
            "reference_strength": float(
                getattr(model, "reference_strength", 0.0)
            ),
            "reference_residual_active": bool(
                getattr(model, "reference_residual_active", False)
            ),
            "semantic_strength": float(getattr(model, "semantic_strength", 0.0)),
            "fsra_strength": float(getattr(model, "fsra_strength", 0.0)),
            "fsra_control": stats.get("_fsra_control"),
            "fsra_descriptions": stats.get("_fsra_descriptions"),
            "fsra_role_names": stats.get("_fsra_role_names"),
            "fsra_transfer_loaded_from": str(args.fsra_load) if args.fsra_load else None,
            "fsra_transfer_source_metadata": fsra_loaded_metadata,
            "fsra_target_semantic_training": bool(fsra_tune and not args.fsra_eval_only),
            "probabilistic_output": bool(getattr(model, "probabilistic_output", False)),
            "probabilistic_state_conditioning": bool(
                getattr(model, "probabilistic_state", False)
                if getattr(model, "probabilistic_output", False) else False
            ),
            "uncertainty_validation_calibration": float(
                getattr(model, "uncertainty_calibration", torch.tensor(1.0))
                .detach().cpu().item()
            ),
            "quantile_output": bool(getattr(model, "quantile_output", False)),
            "quantile_state_conditioning": bool(
                getattr(model, "quantile_state", False)
                if getattr(model, "quantile_output", False) else False
            ),
            "quantile_interval_adjustment_standardized": float(
                getattr(model, "quantile_interval_adjustment", torch.tensor(0.0))
                .detach().cpu().item()
            ),
            "forecast_start_hour": stats.get("forecast_start_hour"),
            "history_features_used": getattr(model, "history_features_used", stats.get("feature_names")),
            "protocol_note": getattr(model, "protocol_note", None)}
    return clean_current_ours_result(result)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--target-col", default="auto")
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", default=["persistence", "seasonal_naive", "dlinear", "patchtst", "itransformer", "ampdnet", "pstf"])
    ap.add_argument("--seq-len", type=int, default=96)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--cycle-len", type=int, default=96)
    ap.add_argument("--patch-len", type=int, default=16)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=32)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument(
        "--ours-patch-hours", nargs="+", type=float,
        default=[1.0, 2.0, 4.0, 8.0],
        help="physical patch durations used only by Ours",
    )
    ap.add_argument(
        "--ours-gtr-period", type=int, default=24,
        help="local/global convolution support in the source GTR block",
    )
    ap.add_argument(
        "--ours-revin-mode", choices=["full", "center", "global"],
        default="full",
        help="window normalization used only by Ours before the shared branches",
    )
    ap.add_argument(
        "--ours-dropout", type=float, default=0.10,
        help="dropout shared by Ours' GTR, CorPatch and unified head",
    )
    ap.add_argument(
        "--ours-corpatch-heads", type=int, default=4,
        help="attention heads in Variable-aware CorPatch",
    )
    ap.add_argument(
        "--ours-cma-heads", type=int, default=4,
        help="independent GPT-2 cross-modal alignment heads used only by Ours",
    )
    ap.add_argument(
        "--ours-cma-dropout", type=float, default=0.05,
        help="dropout in Ours' prompt encoder and multi-head CMA stage",
    )
    ap.add_argument(
        "--ours-cma-training", choices=["staged", "joint"], default="staged",
        help=(
            "fit CMA after a frozen backbone (staged) or jointly with the "
            "time-frequency backbone before optional CMA-only refinement"
        ),
    )
    ap.add_argument(
        "--ours-cma-scale-route", action="store_true",
        help=(
            "add the legacy raw-Prompt scale-routing bypass alongside CMA; "
            "disabled by default so CMA attribution remains explicit"
        ),
    )
    ap.add_argument(
        "--ours-cma-prompt-features",
        choices=[
            "gpt2", "gpt2_numeric", "gpt2_sequence",
            "random_gpt2_sequence", "numeric_sequence",
        ],
        default="gpt2",
        help=(
            "select compact GPT-2 regimes, regimes plus statistics, a full "
            "history GPT-2 prompt, its randomly initialized GPT-2 control, "
            "or an information-matched direct numerical MLP control"
        ),
    )
    ap.add_argument(
        "--ours-cma-epochs", type=int, default=0,
        help="Ours CMA-only epochs; zero preserves the legacy automatic budget",
    )
    ap.add_argument(
        "--ours-cma-lr-multiplier", type=float, default=1.0,
        help="CMA-stage learning rate relative to Ours' numerical backbone rate",
    )
    ap.add_argument(
        "--ours-semantic-strength-max", type=float, default=1.5,
        help="largest validation-calibrated gain for all dynamic Prompt paths",
    )
    ap.add_argument(
        "--ours-semantic-strength-steps", type=int, default=16,
        help="number of validation points in Ours' Prompt gain grid",
    )
    ap.add_argument(
        "--ours-solar-anchor", action="store_true",
        help="enable the optional deterministic solar residual anchor",
    )
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--optimizer", choices=["adam", "adamw"], default="adam",
                    help="optimizer for the deterministic backbone")
    ap.add_argument("--base-scheduler",
                    choices=["none", "cosine", "warmup_cosine"], default="none",
                    help="optional per-step cosine schedule for the deterministic backbone")
    ap.add_argument("--warmup-ratio", type=float, default=0.10,
                    help="fraction of deterministic training steps used for LR warmup")
    ap.add_argument("--checkpoint-average-k", type=int, default=1,
                    help="average the top-k numerical-backbone checkpoints, with validation fallback")
    ap.add_argument("--selection-metric",
                    choices=["all_mse", "daylight_mse", "daylight_combo"],
                    default="all_mse",
                    help="validation-only metric used for checkpoints and residual-strength calibration")
    ap.add_argument("--calibration-metric",
                    choices=["inherit", "all_mse", "daylight_mse", "daylight_combo"],
                    default="inherit",
                    help="optional metric used only for final residual-scale calibration")
    ap.add_argument("--daylight-loss-weight", type=float, default=0.0,
                    help="extra mean-normalized point-loss weight above 1%% of training capacity")
    ap.add_argument("--ramp-loss-weight", type=float, default=0.05,
                    help="coefficient of the capacity-normalized ramp-error loss")
    ap.add_argument("--router-activation", choices=["gelu", "relu"], default="gelu",
                    help="activation used only in physical routing MLPs")
    ap.add_argument("--expert-drop-path", type=float, default=0.0,
                    help="stochastic depth probability for temporal/spectral residual increments")
    ap.add_argument("--gtr-dropout", type=float, default=0.10,
                    help="elementwise dropout inside the periodic GTR residual")
    ap.add_argument(
        "--masked-pretrain-epochs", type=int, default=0,
        help=(
            "training-only masked-history reconstruction epochs; the temporary "
            "decoder is discarded before forecast fine-tuning"
        ),
    )
    ap.add_argument(
        "--masked-pretrain-lr", type=float, default=5e-4,
        help="learning rate for training-only masked-history reconstruction",
    )
    ap.add_argument(
        "--masked-pretrain-delta-weight", type=float, default=0.20,
        help="weight of first-difference reconstruction inside masked spans",
    )
    ap.add_argument("--validation-only", action="store_true",
                    help="skip every test evaluation during hyperparameter screening")
    ap.add_argument(
        "--skip-adapter-stages", action="store_true",
        help=(
            "fit and evaluate only the numerical time-frequency backbone; "
            "FSRA and Prompt/CMA stay exactly disabled"
        ),
    )
    ap.add_argument(
        "--evaluation-projection",
        choices=["none", "nonnegative", "nonnegative_night"],
        default="none",
        help=(
            "target-free inference projection; nonnegative_night additionally "
            "uses only audited deterministic future solar geometry"
        ),
    )
    ap.add_argument("--route-loss-weight", type=float, default=0.02)
    ap.add_argument("--state-loss-weight", type=float, default=0.02)
    ap.add_argument("--router-specialization-weight", type=float, default=0.01)
    ap.add_argument("--candidate-loss-weight", type=float, default=0.1)
    ap.add_argument("--candidate-aux", action="store_true",
                    help="supervise the ungated candidate forecast to prevent output-gate collapse")
    ap.add_argument("--output-correction-floor", type=float, default=0.0,
                    help="minimum contribution of the learned candidate in anchored output fusion")
    ap.add_argument("--output-router-bias", type=float, default=None,
                    help="optional initial bias of the anchored-output correction router")
    ap.add_argument("--anchor-output-mode", choices=["gated", "residual", "direct"],
                    default="gated",
                    help="final interface between the physical reference and learned forecast")
    ap.add_argument("--calibrate-output-scale", action="store_true",
                    help="select one global correction multiplier on validation MSE only")
    ap.add_argument("--local-token-refiner", action="store_true",
                    help="add a small dilated residual refiner after CorPatch encoding")
    ap.add_argument("--local-refiner-dropout", type=float, default=0.05)
    ap.add_argument(
        "--scale-token-mixer", action="store_true",
        help="refine token-time dynamics independently inside every patch scale",
    )
    ap.add_argument(
        "--fixed-scale-token-mixer", action="store_true",
        help="use equal fixed dilation weights instead of learnable aggregation",
    )
    ap.add_argument("--scale-token-mixer-dropout", type=float, default=0.05)
    ap.add_argument("--fsra-loss-weight", type=float, default=0.01)
    ap.add_argument("--fsra-mae-weight", type=float, default=0.20)
    ap.add_argument(
        "--semantic-mae-weight", type=float, default=0.20,
        help="masked MAE auxiliary used only while fitting the frozen-LLM semantic adapter",
    )
    ap.add_argument(
        "--loss-kind-override",
        choices=[
            "mse", "mse_ramp", "mse_phys", "mse_phys_night",
            "mse_ramp_phys", "mse_ramp_phys_night",
            "mse_ramp_phys_night_decomp", "edal_ramp_phys_night",
            "mae", "mae_ramp", "mae_ramp_phys",
            "mse_mae", "mse_mae_ramp", "mse_mae_ramp_phys",
        ],
        default=None,
        help="controlled loss ablation without changing the forecasting graph",
    )
    ap.add_argument("--fsra-load", default=None,
                    help="load a site-independent FSRA adapter checkpoint")
    ap.add_argument("--fsra-save", default=None,
                    help="save only site-independent FSRA adapter parameters")
    ap.add_argument("--fsra-eval-only", action="store_true",
                    help="train the numerical target backbone but never fit FSRA on target data")
    ap.add_argument("--fsra-eval-strength", type=float, default=1.0,
                    help="fixed source-chosen strength for target-only FSRA evaluation")
    ap.add_argument("--max-windows", type=int, default=6000)
    ap.add_argument(
        "--max-train-windows", type=int, default=None,
        help=("optional training-only window cap; validation/test retain the "
              "sampling budget defined by --max-windows"),
    )
    ap.add_argument(
        "--max-val-windows", type=int, default=None,
        help="validation-window cap; 0 evaluates the complete validation split",
    )
    ap.add_argument(
        "--max-test-windows", type=int, default=None,
        help="test-window cap; 0 evaluates the complete chronological test split",
    )
    ap.add_argument(
        "--split-train-fraction", type=float, default=0.70,
        help="chronological end fraction of the training split",
    )
    ap.add_argument(
        "--split-validation-end-fraction", type=float, default=0.85,
        help="chronological end fraction of the validation split",
    )
    ap.add_argument(
        "--train-sampling-policy",
        choices=["uniform", "ramp_enriched", "epoch_uniform"],
        default="uniform",
        help=(
            "training-only window sampler; epoch_uniform rotates one "
            "chronology-stratified fixed-budget subset per epoch"
        ),
    )
    ap.add_argument(
        "--ramp-enrichment-fraction", type=float, default=0.30,
        help="fraction of training windows drawn from high historical-ramp regimes",
    )
    ap.add_argument("--observed-policy", choices=["all", "any", "none"], default="all")
    ap.add_argument("--future-feature-policy", choices=["none", "gefcom_nwp", "gefcom_nwp_core"], default="none",
                    help="strict provenance allowlist for future covariates")
    ap.add_argument(
        "--prompt-future-solar-policy",
        choices=["deterministic_solar_geometry", "none"],
        default="deterministic_solar_geometry",
        help=(
            "whether semantic prompts may include deterministic solar geometry "
            "from the forecast interval; use none for a strict history-only benchmark"
        ),
    )
    ap.add_argument("--forecast-start-hour", type=int, default=None,
                    help="keep only forecasts whose first valid timestamp has this hour")
    ap.add_argument("--rated-capacity", type=float, default=None,
                    help="site rated power in the same unit as target; otherwise training maximum is only an empirical bound")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    if args.checkpoint_average_k < 1:
        ap.error("--checkpoint-average-k must be at least 1")
    if not 0.0 <= args.expert_drop_path < 1.0:
        ap.error("--expert-drop-path must be in [0, 1)")
    if not 0.0 <= args.gtr_dropout < 1.0:
        ap.error("--gtr-dropout must be in [0, 1)")
    if not args.ours_patch_hours or any(value <= 0.0 for value in args.ours_patch_hours):
        ap.error("--ours-patch-hours must contain positive durations")
    if args.ours_gtr_period < 2:
        ap.error("--ours-gtr-period must be at least 2")
    if not 0.0 <= args.ours_dropout < 1.0:
        ap.error("--ours-dropout must be in [0, 1)")
    if args.ours_corpatch_heads < 1 or args.d_model % args.ours_corpatch_heads != 0:
        ap.error("--ours-corpatch-heads must divide --d-model")
    if args.ours_cma_heads < 1:
        ap.error("--ours-cma-heads must be at least 1")
    if not 0.0 <= args.ours_cma_dropout < 1.0:
        ap.error("--ours-cma-dropout must be in [0, 1)")
    if args.ours_cma_epochs < 0:
        ap.error("--ours-cma-epochs must be non-negative")
    if args.ours_cma_lr_multiplier <= 0.0:
        ap.error("--ours-cma-lr-multiplier must be positive")
    if args.ours_semantic_strength_max < 0.0:
        ap.error("--ours-semantic-strength-max must be non-negative")
    if args.ours_semantic_strength_steps < 2:
        ap.error("--ours-semantic-strength-steps must be at least 2")
    if args.masked_pretrain_epochs < 0:
        ap.error("--masked-pretrain-epochs must be non-negative")
    if args.masked_pretrain_lr <= 0.0:
        ap.error("--masked-pretrain-lr must be positive")
    if args.masked_pretrain_delta_weight < 0.0:
        ap.error("--masked-pretrain-delta-weight must be non-negative")
    if args.max_train_windows is not None and args.max_train_windows < 1:
        ap.error("--max-train-windows must be at least 1")
    if args.max_val_windows is not None and args.max_val_windows < 0:
        ap.error("--max-val-windows must be non-negative; 0 means all")
    if args.max_test_windows is not None and args.max_test_windows < 0:
        ap.error("--max-test-windows must be non-negative; 0 means all")
    if not 0.0 < args.split_train_fraction < args.split_validation_end_fraction < 1.0:
        ap.error(
            "split fractions must satisfy "
            "0 < split-train-fraction < split-validation-end-fraction < 1"
        )
    if not 0.0 <= args.ramp_enrichment_fraction < 1.0:
        ap.error("--ramp-enrichment-fraction must be in [0, 1)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_windows(args.data, args.seq_len, args.horizon, args.max_windows,
                        args.target_col, args.cycle_len, args.observed_policy,
                        args.rated_capacity, args.future_feature_policy,
                        args.forecast_start_hour, args.max_train_windows,
                        args.split_train_fraction,
                        args.split_validation_end_fraction,
                        args.train_sampling_policy,
                        args.ramp_enrichment_fraction,
                        args.max_val_windows,
                        args.max_test_windows)
    results = []
    for name in args.models:
        seed_all(args.seed)
        print(f"running model={name} L={args.seq_len} seed={args.seed}", flush=True)
        result = run_one(name, args, data, device)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
