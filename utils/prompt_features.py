"""Leakage-safe offline prompt features for PV forecasting.

The language model is never part of the forecasting graph.  Every dynamic
prompt is constructed from one historical window, encoded offline and cached.
The module supports both compact operating-regime prompts and information-
matched full-history prompts used to distinguish pretrained-language features
from a direct numerical encoder.  No future target or held-out statistic enters
a prompt.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def _regime_statistics(window: np.ndarray, stats: dict) -> dict[str, float]:
    """Return history-only statistics used to form categorical PV prompts."""
    target = np.asarray(window[:, -1], dtype=np.float32)
    dx = np.diff(target)
    names = list(stats.get("feature_names") or [])
    solar_idx = next((names.index(n) for n in
                      ("ClearSkyGHIProxy", "SolarGeometryProxy", "ClearSkyProxy",
                       "SolarZenithCos") if n in names), None)
    if solar_idx is None:
        daylight = float(np.mean(target > 0.0))
        solar_mean = 0.0
    else:
        solar = np.asarray(window[:, solar_idx], dtype=np.float32)
        raw = (solar * float(stats["feature_sd"][solar_idx])
               + float(stats["feature_mu"][solar_idx]))
        daylight = float(np.mean(raw > (20.0 if "GHI" in names[solar_idx] else 0.02)))
        solar_mean = float(np.mean(solar))
    qidx = list(stats.get("quality_indices") or [])
    quality = float(np.mean(window[:, qidx])) if qidx else 1.0
    return {
        "abs_ramp": float(np.mean(np.abs(dx))) if len(dx) else 0.0,
        "directionality": (
            abs(float(np.sum(dx))) / (float(np.sum(np.abs(dx))) + 1e-6)
            if len(dx) else 0.0
        ),
        "oscillation": (
            float(np.mean(dx[1:] * dx[:-1] < 0)) if len(dx) > 1 else 0.0
        ),
        "daylight": daylight,
        "solar_mean": solar_mean,
        "quality": quality,
    }


def fit_prompt_regime_thresholds(x: np.ndarray, train_starts: Iterable[int],
                                 stats: dict) -> dict[str, tuple[float, float]]:
    """Fit categorical-prompt thresholds on training windows only.

    The returned thresholds are frozen before validation/test prompt creation.
    This makes a regime label mean "low/middle/high relative to this site's
    training distribution" without reading any future target or held-out
    statistic.
    """
    seq_len = int(stats.get("seq_len", 96))
    rows = [
        _regime_statistics(x[int(s):int(s) + seq_len], stats)
        for s in train_starts
    ]
    if not rows:
        raise ValueError("cannot fit prompt thresholds without training windows")
    thresholds = {}
    for key in rows[0]:
        selected = rows
        if key in {"abs_ramp", "directionality", "oscillation"}:
            # Night windows are legitimately flat but would otherwise make
            # both dynamic quantiles exactly zero.  Fit dynamic boundaries on
            # windows containing daylight; the fixed night regime remains
            # represented by the solar-state prompt.
            selected = [row for row in rows if row["daylight"] > 0.10]
            if key in {"directionality", "oscillation"}:
                changing = [
                    row for row in selected if row["abs_ramp"] > 1e-6
                ]
                selected = changing or selected
        values = np.asarray([row[key] for row in (selected or rows)], dtype=np.float32)
        low, high = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
        thresholds[key] = (float(low), float(high))
    # Darkness is a physical condition, not a site-relative tercile.
    thresholds["daylight"] = (
        0.10, float(thresholds["daylight"][1])
    )
    return thresholds


def _level(value: float, low: float, high: float) -> str:
    return "low" if value < low else "high" if value > high else "moderate"


def _window_tokens(window: np.ndarray, stats: dict, horizon: int,
                   future_solar: np.ndarray | None = None,
                   version: int = 1) -> tuple[list[str], np.ndarray]:
    dt = float(stats["dt_hours"])
    target = np.asarray(window[:, -1], dtype=np.float32)
    dx = np.diff(target)
    target_mean = float(np.mean(target))
    target_std = float(np.std(target))
    ramp = float(np.mean(np.abs(dx))) if len(dx) else 0.0
    oscillation = float(np.mean(dx[1:] * dx[:-1] < 0)) if len(dx) > 1 else 0.0

    names = list(stats.get("feature_names") or [])
    solar_idx = next((names.index(n) for n in
                      ("ClearSkyGHIProxy", "SolarGeometryProxy", "ClearSkyProxy",
                       "SolarZenithCos") if n in names), None)
    if solar_idx is None:
        solar = np.zeros(len(window), dtype=np.float32)
        daylight = float(np.mean(target > 0.0))
    else:
        solar = np.asarray(window[:, solar_idx], dtype=np.float32)
        # Features are standardized with training statistics.  Zero is the
        # training mean, so this is a relative state descriptor, not an oracle.
        raw = solar * float(stats["feature_sd"][solar_idx]) + float(stats["feature_mu"][solar_idx])
        daylight = float(np.mean(raw > (20.0 if "GHI" in names[solar_idx] else 0.02)))
    solar_mean = float(np.mean(solar))

    qidx = list(stats.get("quality_indices") or [])
    quality = float(np.mean(window[:, qidx])) if qidx else 1.0
    history_hours = len(window) * dt
    forecast_hours = horizon * dt
    phase = "rising" if len(solar) > 1 and solar[-1] - solar[0] > 0.25 else (
        "setting" if len(solar) > 1 and solar[-1] - solar[0] < -0.25 else "mixed")

    texts = [
        (f"Photovoltaic forecasting task: use {history_hours:g} hours of history "
         f"sampled every {dt * 60:g} minutes to predict the next {forecast_hours:g} hours."),
        (f"Historical solar operating state: {_level(daylight, 0.25, 0.7)} daylight "
         f"exposure, {_level(solar_mean, -0.5, 0.5)} solar input, {phase} solar phase."),
        (f"Historical photovoltaic response: {_level(target_mean, -0.5, 0.5)} power "
         f"level and {_level(target_std, 0.5, 1.0)} variability."),
        (f"Historical disturbance state: {_level(ramp, 0.15, 0.5)} ramp activity, "
         f"{_level(oscillation, 0.2, 0.5)} oscillation, {_level(quality, 0.8, 0.98)} data quality."),
    ]
    numeric = np.asarray([
        [min(history_hours / 180.0, 1.0), min(forecast_hours / 180.0, 1.0)],
        [daylight, np.clip(solar_mean / 3.0, -1.0, 1.0)],
        [np.clip(target_mean / 3.0, -1.0, 1.0), min(target_std / 3.0, 1.0)],
        [min(ramp / 3.0, 1.0), oscillation],
    ], dtype=np.float32)
    if version >= 2:
        # Quantitative language carries substantially more sample information
        # than four coarse bins. Values are in the training-standardized space
        # and rounded only to limit the number of offline GPT-2 calls.
        max_ramp = float(np.max(np.abs(dx))) if len(dx) else 0.0
        net_change = float(target[-1] - target[0]) if len(target) else 0.0
        weather_idx = [i for i, name in enumerate(names[:-1]) if any(key in name.lower() for key in (
            "irradi", "radiation", "temperature", "humidity", "cloud", "wind",
        ))][:6]
        weather_desc = ", ".join(
            f"{names[i]} mean {float(np.mean(window[:, i])):.1f}"
            for i in weather_idx
        ) or "no additional measured weather variable"
        texts = [
            texts[0],
            (f"Historical normalized photovoltaic power has mean {target_mean:.1f}, standard deviation "
             f"{target_std:.1f}, last value {float(target[-1]):.1f}, and maximum {float(target.max()):.1f}."),
            (f"Historical photovoltaic transitions have mean absolute ramp {ramp:.1f}, maximum ramp "
             f"{max_ramp:.1f}, net change {net_change:.1f}, and sign reversal fraction {oscillation:.1f}."),
            (f"Historical solar state has daylight fraction {daylight:.1f}, normalized solar mean "
             f"{solar_mean:.1f}, and {phase} phase."),
            f"Historical measured weather summary: {weather_desc}.",
            f"Historical observation quality fraction is {quality:.1f}.",
        ]
        if future_solar is not None and len(future_solar):
            for j, part in enumerate(np.array_split(np.asarray(future_solar), 4), start=1):
                if len(part):
                    texts.append(
                        f"Known deterministic future solar quarter {j} has normalized mean "
                        f"{float(np.mean(part)):.1f}, maximum {float(np.max(part)):.1f}, and daylight "
                        f"fraction {float(np.mean(part > 0.02)):.1f}."
                    )
    if version >= 3:
        # Official T3Time granularity: one prompt and one frozen GPT-2
        # last-token embedding for every input variable.  Decimal values are
        # retained because the PV benchmark stores training-standardized data.
        texts = []
        for idx in range(len(names)):
            values = np.asarray(window[:, idx], dtype=np.float32)
            ordered = ", ".join(f"{float(v):.2f}" for v in values)
            trend = float(np.sum(np.diff(values))) if len(values) > 1 else 0.0
            texts.append(
                f"During the historical window, the standardized {names[idx]} values were "
                f"{ordered} every {dt * 60:g} minutes. The total trend value was {trend:.2f}."
            )
    if version >= 5:
        # v5 deliberately uses a small, finite vocabulary of operating-regime
        # prototypes.  GPT-2 is a language model, not a numerical encoder: a
        # prompt containing 96 decimal values is both vulnerable to 512-token
        # truncation and nearly collinear after final-token pooling.  These
        # prototypes preserve the physically meaningful distinctions while
        # making the semantic intervention falsifiable by shuffle/zero/random
        # controls.  Every category is computed from history or deterministic
        # future solar geometry; no future measured PV/weather enters here.
        abs_ramp = float(np.mean(np.abs(dx))) if len(dx) else 0.0
        net_change = float(target[-1] - target[0]) if len(target) else 0.0
        directionality = (abs(float(np.sum(dx))) /
                          (float(np.sum(np.abs(dx))) + 1e-6)) if len(dx) else 0.0

        regime_thresholds = stats.get("prompt_regime_thresholds") or {}

        def bounds(key: str, default: tuple[float, float]) -> tuple[float, float]:
            value = regime_thresholds.get(key, default)
            return float(value[0]), float(value[1])

        ramp_low, _ = bounds("abs_ramp", (0.10, 0.50))
        _, direction_high = bounds("directionality", (0.30, 0.60))
        oscillation_low, oscillation_high = bounds("oscillation", (0.20, 0.45))
        daylight_low, daylight_high = bounds("daylight", (0.10, 0.45))
        _, solar_high = bounds("solar_mean", (-0.50, 0.50))
        quality_low, quality_high = bounds("quality", (0.80, 0.98))

        if abs_ramp <= ramp_low:
            response = "stable photovoltaic output with little short-term change"
        elif directionality >= direction_high and net_change > 0:
            response = "a sustained upward photovoltaic ramp"
        elif directionality >= direction_high and net_change < 0:
            response = "a sustained downward photovoltaic ramp"
        elif oscillation >= oscillation_high:
            response = "rapid alternating photovoltaic fluctuations consistent with fragmented-cloud disturbance"
        else:
            response = "irregular photovoltaic variation without a dominant ramp direction"

        if daylight < daylight_low:
            solar_regime = "nighttime with negligible solar potential"
        elif daylight < daylight_high:
            solar_regime = "a sunrise or sunset transition with partial daylight"
        elif solar_mean > solar_high:
            solar_regime = "strong daytime solar potential"
        else:
            solar_regime = "mixed daytime solar potential"

        if oscillation < oscillation_low:
            disturbance = "low oscillatory disturbance"
        elif oscillation < oscillation_high:
            disturbance = "moderate oscillatory disturbance"
        else:
            disturbance = "high oscillatory disturbance"

        if quality >= quality_high:
            quality_regime = "highly reliable historical observations"
        elif quality >= quality_low:
            quality_regime = "partially reliable historical observations"
        else:
            quality_regime = "low-quality historical observations requiring conservative correction"

        if forecast_hours <= 1.0:
            horizon_regime = "very-short-term photovoltaic forecasting"
        elif forecast_hours <= 8.0:
            horizon_regime = "intra-day photovoltaic forecasting"
        elif forecast_hours <= 24.0:
            horizon_regime = "day-ahead photovoltaic forecasting"
        else:
            horizon_regime = "multi-day photovoltaic forecasting"

        if future_solar is None or len(future_solar) == 0:
            future_regime = "deterministic future solar geometry is unavailable"
        else:
            future_solar = np.asarray(future_solar, dtype=np.float32)
            future_daylight = float(np.mean(future_solar > 0.02))
            if future_daylight < 0.10:
                future_regime = "the forecast horizon is almost entirely nighttime"
            elif future_daylight < 0.55:
                future_regime = "the forecast horizon crosses a daylight boundary"
            else:
                future_regime = "the forecast horizon contains sustained daylight"

        texts = [
            f"Task context: {horizon_regime} from historical measurements.",
            f"Historical solar operating regime: {solar_regime}.",
            f"Historical power response regime: {response}.",
            f"Historical spectral disturbance regime: {disturbance}.",
            f"Observation regime: {quality_regime}.",
            f"Known solar-geometry context: {future_regime}.",
        ]
        if version >= 8:
            # The task/horizon sentence is constant for a fixed experimental
            # setting.  Removing it prevents GPT-2's common language direction
            # from behaving as an extra learned bias rather than sample-level
            # semantic evidence.
            texts = texts[1:]
    if version >= 6:
        # v6 is a genuinely sample-dependent, scale-matched prompt bank.  It
        # describes three causal history summaries whose physical support
        # matches the 2 h / 4 h / 8 h numerical patches used by PCARR-v15.
        # The language model receives categorical/rounded physical quantities,
        # not a 96-value decimal dump.  This preserves GPT-2's semantic role,
        # keeps the vocabulary finite enough for offline caching, and prevents
        # prompt truncation from silently discarding the scale summary.
        capacity = max(float(stats.get("capacity", 1.0)), 1e-6)
        target_mu = float(stats.get("target_mu", 0.0))
        target_sd = max(float(stats.get("target_sd", 1.0)), 1e-6)
        target_phys = target * target_sd + target_mu
        power_norm = target_phys / capacity

        clear_raw = None
        if solar_idx is not None:
            clear_raw = (np.asarray(window[:, solar_idx], dtype=np.float32)
                         * float(stats["feature_sd"][solar_idx])
                         + float(stats["feature_mu"][solar_idx]))
        clear_scale = max(float(stats.get("clear_sky_training_max") or 1.0), 1e-6)

        def category(value: float, low: float, high: float,
                     labels: tuple[str, str, str]) -> str:
            return labels[0] if value < low else labels[2] if value > high else labels[1]

        def scale_summary(hours: float, scale_name: str) -> tuple[str, list[float]]:
            steps = min(len(window), max(2, int(round(hours / max(dt, 1e-6)))))
            pv = power_norm[-steps:]
            dp = np.diff(pv)
            ramp_rate = (float(np.mean(np.abs(dp))) / max(dt, 1e-6)
                         if len(dp) else 0.0)
            net_rate = (float(pv[-1] - pv[0]) / max((steps - 1) * dt, dt)
                        if len(pv) > 1 else 0.0)
            variability = float(np.std(pv))
            reversal = (float(np.mean(dp[1:] * dp[:-1] < 0.0))
                        if len(dp) > 1 else 0.0)
            directionality = (abs(float(np.sum(dp))) /
                              (float(np.sum(np.abs(dp))) + 1e-6)
                              if len(dp) else 0.0)
            if ramp_rate < 0.03:
                response = "stable photovoltaic response"
            elif directionality > 0.60 and net_rate > 0.0:
                response = "persistent upward photovoltaic ramp"
            elif directionality > 0.60 and net_rate < 0.0:
                response = "persistent downward photovoltaic ramp"
            elif reversal > 0.40:
                response = "alternating photovoltaic disturbance"
            else:
                response = "irregular photovoltaic transition"

            variability_state = category(
                variability, 0.04, 0.15,
                ("low variability", "moderate variability", "high variability"),
            )
            oscillation_state = category(
                reversal, 0.20, 0.45,
                ("weak oscillation", "moderate oscillation", "strong oscillation"),
            )
            quality_state = category(
                quality, 0.80, 0.98,
                ("low observation reliability", "partial observation reliability",
                 "high observation reliability"),
            )

            if clear_raw is None:
                solar_state = "solar geometry unavailable"
                daylight_local = float(np.mean(pv > 0.01))
                solar_phase_local = "unknown solar phase"
            else:
                local_clear = clear_raw[-steps:]
                clear_norm = local_clear / clear_scale
                threshold = 0.02
                daylight_local = float(np.mean(clear_norm > threshold))
                delta_clear = float(clear_norm[-1] - clear_norm[0])
                solar_phase_local = (
                    "rising solar phase" if delta_clear > 0.10 else
                    "setting solar phase" if delta_clear < -0.10 else
                    "nearly stationary solar phase"
                )
                performance = pv / np.maximum(clear_norm, 0.02)
                valid = clear_norm > threshold
                performance_level = float(np.median(performance[valid])) if valid.any() else 0.0
                solar_state = category(
                    performance_level, 0.45, 0.90,
                    ("strong clear-sky attenuation", "moderate clear-sky attenuation",
                     "near clear-sky response"),
                )

            prompt = (
                f"Photovoltaic forecasting scale: {scale_name}, covering the latest {hours:g} hours. "
                f"Forecast horizon: {forecast_hours:g} hours. "
                f"Historical power level: {category(float(np.mean(pv)), 0.15, 0.70, ('low', 'moderate', 'high'))}. "
                f"Power dynamics: {response}, {variability_state}, and {oscillation_state}. "
                f"Solar context: {solar_phase_local} with {solar_state}. "
                f"Observation context: {quality_state}. "
                f"Summary state: {scale_name.upper()} | {response.upper()} | {solar_phase_local.upper()}."
            )
            numeric_row = [
                float(np.mean(pv)), variability, ramp_rate, net_rate,
                reversal, daylight_local, quality, min(forecast_hours / 180.0, 1.0),
            ]
            return prompt, numeric_row

        scale_specs = (
            (2.0, "short-scale cloud and ramp response"),
            (4.0, "medium-scale weather transition"),
            (8.0, "long-scale solar and daily-shape context"),
        )
        scale_pairs = [scale_summary(hours, name) for hours, name in scale_specs]
        texts = [item[0] for item in scale_pairs]
        numeric = np.asarray([item[1] for item in scale_pairs], dtype=np.float32)
    if version == 9:
        # Efficient variable-wise semantic prompt for the post-fusion adapter.
        # Keep T3Time's one-prompt-per-variable interface, but describe finite
        # history-only operating regimes instead of asking GPT-2 to encode 96
        # decimal values.  This is both semantically better matched to a frozen
        # language model and makes offline caching practical for full datasets.
        texts = []
        numeric_rows = []

        def regime(value: float, low: float, high: float) -> str:
            return "low" if value < low else "high" if value > high else "moderate"

        for idx, name in enumerate(names):
            values = np.asarray(window[:, idx], dtype=np.float32)
            level = float(np.mean(values))
            variability = float(np.std(values))
            change = float(values[-1] - values[0]) if len(values) > 1 else 0.0
            direction = (
                "falling" if change < -0.5 else
                "rising" if change > 0.5 else
                "stable"
            )
            level_state = regime(level, -0.5, 0.5)
            variability_state = regime(variability, 0.5, 1.2)
            texts.append(
                f"Historical variable {name} is in a {level_state} standardized level, "
                f"with {variability_state} variability and a {direction} trend. "
                f"Use this history-only operating regime for {forecast_hours:g}-hour "
                "photovoltaic forecasting."
            )
            numeric_rows.append([
                np.clip(level / 3.0, -1.0, 1.0),
                np.clip(variability / 3.0, 0.0, 1.0),
                np.clip(change / 3.0, -1.0, 1.0),
            ])
        numeric = np.asarray(numeric_rows, dtype=np.float32)
    if version == 10:
        # High-information, T3Time-style prompt.  Unlike v9, this retains the
        # ordered history for every variable.  Values are quantized training-
        # standardized measurements: this keeps all 96 positions while staying
        # comfortably below GPT-2's 512-token limit.  The direct-numeric control
        # receives the same ordered values plus the same history-only summaries.
        texts = []
        numeric_rows = []
        feature_mu = np.asarray(stats["feature_mu"], dtype=np.float32)
        feature_sd = np.asarray(stats["feature_sd"], dtype=np.float32)

        def physical_role(name: str, index: int) -> str:
            normalized = name.lower()
            if index == len(names) - 1 or normalized == "target":
                return "photovoltaic power response"
            if any(term in normalized for term in (
                    "radiation", "irradiance", "ghi", "dhi", "dni", "solar")):
                return "solar-energy or solar-geometry driver"
            if any(term in normalized for term in (
                    "temperature", "humidity", "cloud", "wind", "rain")):
                return "historical weather driver"
            if any(term in normalized for term in ("clock", "day", "hour")):
                return "deterministic historical time coordinate"
            if any(term in normalized for term in ("quality", "observed", "mask")):
                return "historical observation-quality indicator"
            return "historical measured covariate"

        for idx, name in enumerate(names):
            values = np.asarray(window[:, idx], dtype=np.float32)
            clipped = np.clip(values, -8.0, 8.0)
            # One integer represents 0.1 training standard deviations.  This
            # is much less lossy than regime labels but avoids fragile decimal
            # tokenization and silent truncation.
            quantized = np.rint(10.0 * clipped).astype(np.int16)
            ordered = " ".join(str(int(value)) for value in quantized)
            delta = np.diff(values)
            level = float(np.mean(values))
            variability = float(np.std(values))
            last = float(values[-1])
            trend = float(values[-1] - values[0]) if len(values) > 1 else 0.0
            ramp = float(np.mean(np.abs(delta))) if len(delta) else 0.0
            max_ramp = float(np.max(np.abs(delta))) if len(delta) else 0.0
            reversal = (
                float(np.mean(delta[1:] * delta[:-1] < 0.0))
                if len(delta) > 1 else 0.0
            )
            physical = values * feature_sd[idx] + feature_mu[idx]
            physical_mean = float(np.mean(physical))
            physical_last = float(physical[-1])
            role = physical_role(name, idx)
            texts.append(
                f"Historical variable: {name}. Physical role: {role}. "
                f"Sampling interval: {dt * 60:g} minutes; history length: {len(values)}. "
                "Ordered standardized history codes, where one code unit is 0.1 training "
                f"standard deviations: {ordered}. "
                f"History-only statistics: standardized mean {level:.2f}, variability "
                f"{variability:.2f}, last value {last:.2f}, net trend {trend:.2f}, mean "
                f"absolute ramp {ramp:.2f}, maximum ramp {max_ramp:.2f}, reversal fraction "
                f"{reversal:.2f}, physical mean {physical_mean:.2f}, and physical last "
                f"value {physical_last:.2f}. Forecast horizon: {forecast_hours:g} hours."
            )
            numeric_rows.append(np.concatenate([
                (clipped / 4.0).astype(np.float32),
                np.asarray([
                    np.clip(level / 3.0, -1.0, 1.0),
                    np.clip(variability / 3.0, 0.0, 1.0),
                    np.clip(last / 3.0, -1.0, 1.0),
                    np.clip(trend / 6.0, -1.0, 1.0),
                    np.clip(ramp / 3.0, 0.0, 1.0),
                    np.clip(max_ramp / 6.0, 0.0, 1.0),
                    reversal,
                    min(forecast_hours / 24.0, 1.0),
                ], dtype=np.float32),
            ]))
        numeric = np.asarray(numeric_rows, dtype=np.float32)
    return texts, numeric


@torch.no_grad()
def _encode_unique(texts: list[str], cache_root: Path, batch_size: int = 8,
                   mean_last: bool = False,
                   model_family: str = "gpt2",
                   random_seed: int = 2026) -> dict[str, np.ndarray]:
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    if model_family == "qwen25_instruct":
        repo = "models--Qwen--Qwen2.5-0.5B-Instruct"
    elif model_family in {"gpt2", "gpt2_random"}:
        repo = "models--gpt2"
    else:
        raise ValueError(f"unsupported frozen language model: {model_family}")
    repo_root = cache_root / "hub" / repo
    if not repo_root.exists():
        repo_root = cache_root / repo
    snapshots = sorted((repo_root / "snapshots").glob("*"))
    if not snapshots:
        raise FileNotFoundError(f"frozen {model_family} snapshot is absent under {cache_root}")
    model_path = str(snapshots[-1])
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_dtype = torch.float16 if device.type == "cuda" else torch.float32
    if model_family == "gpt2_random":
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        # A local fork prevents the control encoder from perturbing forecasting
        # initialization.  It uses the exact GPT-2 architecture and tokenizer,
        # but none of the pretrained language weights.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(random_seed))
            model = AutoModel.from_config(config)
        model = model.to(device=device, dtype=load_dtype)
    else:
        model = AutoModel.from_pretrained(
            model_path, local_files_only=True, dtype=load_dtype,
        ).to(device)
    model.eval()
    result: dict[str, np.ndarray] = {}
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        if model_family == "qwen25_instruct":
            encoded_text = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": (
                            "You are a photovoltaic forecasting expert. Encode the supplied "
                            "historical operating state for time-scale-aware residual fusion."
                        )},
                        {"role": "user", "content": sentence},
                    ],
                    tokenize=False, add_generation_prompt=True,
                )
                for sentence in batch
            ]
        else:
            encoded_text = batch
        tok = tokenizer(
            encoded_text, padding=True, truncation=True, max_length=512,
            return_tensors="pt",
        )
        tok = {k: v.to(device) for k, v in tok.items()}
        hidden = model(**tok).last_hidden_state
        positions = torch.arange(hidden.shape[1], device=device).unsqueeze(0)
        last = (positions * tok["attention_mask"]).argmax(dim=1)
        pooled = hidden[torch.arange(len(batch), device=device), last]
        if mean_last:
            mask = tok["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            mean = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            pooled = torch.cat([mean, pooled], dim=-1)
        for sentence, vector in zip(batch, pooled.float().cpu().numpy()):
            result[sentence] = vector.astype(np.float16)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def build_or_load_prompt_features(x: np.ndarray, starts: Iterable[int], stats: dict,
                                  horizon: int, cache_file: Path,
                                  hf_cache: Path, prompt_version: int = 1,
                                  include_future_solar: bool = True,
                                  encoder_kind: str = "gpt2",
                                  random_seed: int = 2026) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    if encoder_kind not in {"gpt2", "gpt2_random", "numeric"}:
        raise ValueError(f"unsupported prompt encoder: {encoder_kind}")
    starts = np.asarray(sorted(set(int(s) for s in starts)), dtype=np.int64)
    if cache_file.exists():
        saved = np.load(cache_file)
        if np.array_equal(saved["starts"], starts):
            llm = {int(s): v.astype(np.float32) for s, v in zip(starts, saved["llm"])}
            numeric = {int(s): v.astype(np.float32) for s, v in zip(starts, saved["numeric"])}
            return llm, numeric

    all_texts: list[list[str]] = []
    numerics = []
    seq_len = int(stats.get("seq_len", 96))
    names = list(stats.get("feature_names") or [])
    solar_idx = (
        next((names.index(n) for n in
              ("ClearSkyGHIProxy", "SolarGeometryProxy", "ClearSkyProxy",
               "SolarZenithCos") if n in names), None)
        if include_future_solar else None
    )
    for s in starts:
        future_solar = (x[s + seq_len:s + seq_len + horizon, solar_idx]
                        if prompt_version >= 2 and solar_idx is not None else None)
        if future_solar is not None and prompt_version >= 6:
            # v6 prompts may mention deterministic solar geometry in future
            # revisions.  Store it in physical units now so any such feature is
            # provenance-explicit and never confused with a standardized zero.
            future_solar = (future_solar * float(stats["feature_sd"][solar_idx])
                            + float(stats["feature_mu"][solar_idx]))
        texts, numeric = _window_tokens(
            x[s:s + seq_len], stats, horizon, future_solar, prompt_version,
        )
        all_texts.append(texts)
        numerics.append(numeric)
    numeric_array = np.stack(numerics).astype(np.float32)
    if encoder_kind == "numeric":
        # Preserve the return contract without loading a language model.  The
        # caller selects ``numeric_array`` and configures a numerical MLP.
        llm_array = numeric_array.astype(np.float16)
    else:
        unique = sorted({sentence for group in all_texts for sentence in group})
        model_family = (
            "qwen25_instruct" if prompt_version == 7 else
            "gpt2_random" if encoder_kind == "gpt2_random" else
            "gpt2"
        )
        encoded = _encode_unique(
            unique, hf_cache,
            mean_last=(prompt_version >= 4 and prompt_version not in {9, 10}),
            model_family=model_family,
            random_seed=random_seed,
        )
        llm_array = np.stack([
            [encoded[sentence] for sentence in group] for group in all_texts
        ])
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, starts=starts, llm=llm_array, numeric=numeric_array)
    llm = {int(s): v.astype(np.float32) for s, v in zip(starts, llm_array)}
    numeric = {int(s): v for s, v in zip(starts, numeric_array)}
    return llm, numeric
