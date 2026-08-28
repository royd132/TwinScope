"""Leakage-safe frozen text embeddings for PV channel metadata.

Unlike the earlier window prompts, these descriptions contain no numerical
history and no future observations.  They describe only a channel's identity,
unit, availability and physical role, so the frozen language encoder supplies
metadata that is not recoverable from one value window alone.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_TEXT_MODEL = "nomic-ai/nomic-embed-text-v1.5"


# Fixed natural-language anchors used to turn an unrestricted dense language
# embedding into a compact, inspectable physical-role signature.  They are not
# trainable experts and never receive numerical samples.
PHYSICAL_ROLE_PROTOTYPES = {
    "pv_response": (
        "clustering: Historical photovoltaic electrical power output: the "
        "system response and forecast target, constrained by solar resource, "
        "conversion efficiency and equipment limits."
    ),
    "solar_input": (
        "clustering: Measured solar irradiance energy incident on a photovoltaic "
        "array, including global, direct, diffuse or plane-of-array radiation."
    ),
    "thermal_efficiency": (
        "clustering: Module or ambient thermal conditions that modify photovoltaic "
        "conversion efficiency and operating temperature."
    ),
    "atmospheric_regime": (
        "clustering: Atmospheric moisture, pressure, precipitation, visibility or "
        "cloud-related weather state that changes available solar radiation."
    ),
    "weather_motion": (
        "clustering: Wind and atmospheric motion describing cooling or movement of "
        "cloud and weather fields."
    ),
    "solar_geometry": (
        "clustering: Deterministic solar position, daylight availability or a "
        "clear-sky physical envelope known from geometry."
    ),
    "cyclic_time": (
        "clustering: Deterministic cyclic clock or solar-day coordinate locating a "
        "sample within the daily photovoltaic cycle."
    ),
    "data_quality": (
        "clustering: Observation availability, missingness or data-quality metadata; "
        "a reliability indicator rather than an energy input."
    ),
    "accumulated_energy": (
        "clustering: Accumulated photovoltaic or grid energy production rather than "
        "instantaneous power or irradiance."
    ),
}


def _compact_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def describe_channel(name: str) -> str:
    """Return a factual, fixed-schema description for one input channel."""
    key = _compact_name(name)
    label = name.replace("_", " ")
    unit = "unknown or dataset-defined"
    role = "historical measured covariate"
    relation = "its relevance to photovoltaic output must be learned from data"
    availability = "historical observation only"

    if key in {"target", "pvpower", "power", "acpower", "activepower"}:
        label, unit = "photovoltaic active AC power", "kW"
        role = "forecast target and historical photovoltaic system response"
        relation = "it is the output response to solar input, weather and equipment limits"
    elif "targetobserved" in key or "quality" in key or "observedflag" in key:
        label, unit = "target observation quality flag", "dimensionless binary flag"
        role = "indicates whether the historical power value was genuinely observed"
        relation = "it controls reliability and is not a physical energy input"
    elif ("clearsky" not in key
          and any(token in key for token in ("globalhorizontalradiation", "ghi", "globalirradiance"))):
        label, unit = "global horizontal solar irradiance", "W per square metre"
        role = "measured exogenous solar-energy input"
        relation = "it generally increases photovoltaic power during daylight"
    elif any(token in key for token in ("poairradiance", "globaltilted", "radiationglobaltilted")):
        label, unit = "plane-of-array solar irradiance", "W per square metre"
        role = "measured solar-energy input incident on the photovoltaic plane"
        relation = "it generally increases photovoltaic power, with possible clipping at high input"
    elif any(token in key for token in ("diffusehorizontal", "diffusetilted", "dhi")):
        label, unit = "diffuse solar irradiance", "W per square metre"
        role = "diffuse component of the measured solar-energy input"
        relation = "it contributes to photovoltaic power under cloudy and diffuse-light conditions"
    elif "clearsky" in key:
        label, unit = "deterministic clear-sky irradiance proxy", "W per square metre"
        role = "solar-geometry-based reference potential, not future measured weather"
        relation = "it describes the physically available daylight envelope"
        availability = "deterministically known for history and forecast horizon"
    elif "solarzenith" in key or "solargeometry" in key:
        label, unit = "cosine of solar zenith geometry", "dimensionless"
        role = "deterministic solar-position descriptor"
        relation = "positive values indicate daylight solar potential"
        availability = "deterministically known for history and forecast horizon"
    elif "solarelevationmask" in key or "daylight" in key:
        label, unit = "solar elevation daylight mask", "dimensionless binary flag"
        role = "deterministic indicator of whether the sun is above the horizon"
        relation = "photovoltaic power should be near zero outside daylight"
        availability = "deterministically known for history and forecast horizon"
    elif "solardaysin" in key or "clocksin" in key:
        label, unit = "sine component of local solar-day phase", "dimensionless"
        role = "deterministic cyclic time coordinate"
        relation = "it locates the observation within the daily solar cycle"
        availability = "deterministically known"
    elif "solardaycos" in key or "clockcos" in key:
        label, unit = "cosine component of local solar-day phase", "dimensionless"
        role = "deterministic cyclic time coordinate"
        relation = "it locates the observation within the daily solar cycle"
        availability = "deterministically known"
    elif "moduletemperature" in key or "paneltemperature" in key:
        label, unit = "photovoltaic module temperature", "degrees Celsius"
        role = "measured photovoltaic operating-temperature state"
        relation = "higher module temperature generally reduces conversion efficiency"
    elif "temperature" in key or "dewpoint" in key:
        label, unit = label, "degrees Celsius"
        role = "measured atmospheric thermal state"
        relation = "it affects module temperature, efficiency and weather regime"
    elif "humidity" in key:
        label, unit = "relative humidity", "percent"
        role = "measured atmospheric moisture state"
        relation = "it is associated with cloud, haze and weather-regime changes"
    elif "windspeed" in key:
        label, unit = "wind speed", "metres per second"
        role = "measured atmospheric motion and module-cooling covariate"
        relation = "it can affect module cooling and cloud-field evolution"
    elif "winddirection" in key:
        label, unit = label, "degrees or dimensionless directional component"
        role = "measured wind direction descriptor"
        relation = "it describes movement direction of weather systems"
    elif "precip" in key or "rainfall" in key:
        label, unit = "precipitation or rainfall", "millimetres"
        role = "measured adverse-weather indicator"
        relation = "precipitation is commonly associated with reduced solar input"
    elif "pressure" in key:
        label, unit = "surface atmospheric pressure", "hectopascals"
        role = "measured synoptic weather-state covariate"
        relation = "it helps characterize changing weather regimes"
    elif "visibility" in key:
        label, unit = "atmospheric visibility", "kilometres"
        role = "measured atmospheric clarity covariate"
        relation = "reduced visibility can indicate haze, cloud or precipitation"
    elif "energy" in key:
        label, unit = label, "dataset-defined energy unit"
        role = "historical photovoltaic or grid energy measurement"
        relation = "it is related to accumulated electrical production rather than instantaneous irradiance"

    return (
        "clustering: "
        f"Channel name: {label}. Unit: {unit}. Physical role: {role}. "
        f"Relationship to photovoltaic power: {relation}. Availability: {availability}."
    )


def build_channel_descriptions(feature_names: Iterable[str]) -> list[str]:
    return [describe_channel(str(name)) for name in feature_names]


@torch.no_grad()
def _encode_descriptions(descriptions: list[str], cache_root: Path,
                         model_name: str) -> np.ndarray:
    cache_root = Path(cache_root).resolve()
    os.environ.setdefault("HF_MODULES_CACHE", str(cache_root / "modules"))
    from transformers import AutoModel, AutoTokenizer

    local_repo = "models--" + model_name.replace("/", "--")
    snapshots = sorted((Path(cache_root) / local_repo / "snapshots").glob("*"))
    model_source = str(snapshots[-1]) if snapshots else model_name
    local_only = bool(snapshots)
    tokenizer = AutoTokenizer.from_pretrained(
        model_source, cache_dir=str(cache_root), trust_remote_code=True,
        local_files_only=local_only,
    )
    model = AutoModel.from_pretrained(
        model_source, cache_dir=str(cache_root), trust_remote_code=True,
        local_files_only=local_only,
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    encoded = tokenizer(
        descriptions, padding=True, truncation=True, max_length=192,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    hidden = model(**encoded).last_hidden_state
    mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    pooled = F.normalize(pooled.float(), dim=-1)
    return pooled.cpu().numpy().astype(np.float32)


def load_channel_semantics(feature_names: Iterable[str], cache_root: Path,
                           model_name: str = DEFAULT_TEXT_MODEL) -> tuple[np.ndarray, list[str]]:
    """Encode channel descriptions once and cache only the frozen vectors."""
    names = [str(name) for name in feature_names]
    descriptions = build_channel_descriptions(names)
    payload = json.dumps(
        {"model": model_name, "names": names, "descriptions": descriptions},
        ensure_ascii=False, sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:20]
    vector_dir = Path(cache_root) / "channel_semantic_vectors"
    vector_dir.mkdir(parents=True, exist_ok=True)
    vector_file = vector_dir / f"{digest}.npz"
    if vector_file.exists():
        with np.load(vector_file) as loaded:
            vectors = loaded["vectors"].astype(np.float32)
        if vectors.shape[0] != len(names):
            raise ValueError("cached channel semantic vector count is inconsistent")
        return vectors, descriptions

    vectors = _encode_descriptions(descriptions, Path(cache_root), model_name)
    np.savez_compressed(
        vector_file, vectors=vectors,
        names=np.asarray(names), descriptions=np.asarray(descriptions),
    )
    return vectors, descriptions


def _load_role_prototypes(cache_root: Path,
                          model_name: str) -> tuple[np.ndarray, list[str]]:
    """Load frozen embeddings for the fixed physical-role anchors."""
    role_names = list(PHYSICAL_ROLE_PROTOTYPES)
    role_text = [PHYSICAL_ROLE_PROTOTYPES[name] for name in role_names]
    payload = json.dumps(
        {"model": model_name, "roles": PHYSICAL_ROLE_PROTOTYPES},
        ensure_ascii=False, sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:20]
    vector_dir = Path(cache_root) / "channel_semantic_vectors"
    vector_dir.mkdir(parents=True, exist_ok=True)
    vector_file = vector_dir / f"role_prototypes_{digest}.npz"
    if vector_file.exists():
        with np.load(vector_file) as loaded:
            vectors = loaded["vectors"].astype(np.float32)
        return vectors, role_names
    vectors = _encode_descriptions(role_text, Path(cache_root), model_name)
    np.savez_compressed(
        vector_file, vectors=vectors, names=np.asarray(role_names),
        descriptions=np.asarray(role_text),
    )
    return vectors, role_names


def load_channel_role_signatures(
        feature_names: Iterable[str], cache_root: Path,
        model_name: str = DEFAULT_TEXT_MODEL,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Return an interpretable frozen-LM physical-role signature per channel.

    Raw cosine similarities from the sentence encoder are anisotropic.  The
    per-channel centering and scaling below retain relative affinity to the
    physical anchors while removing the shared sentence component.  No target
    samples or dataset statistics participate in this transformation.
    """
    channel_vectors, descriptions = load_channel_semantics(
        feature_names, cache_root, model_name,
    )
    role_vectors, role_names = _load_role_prototypes(cache_root, model_name)
    scores = channel_vectors @ role_vectors.T
    scores = scores - scores.mean(axis=1, keepdims=True)
    scores = scores / np.maximum(scores.std(axis=1, keepdims=True), 1e-6)
    scores = scores / np.maximum(np.linalg.norm(scores, axis=1, keepdims=True), 1e-6)
    return scores.astype(np.float32), descriptions, role_names


def controlled_channel_semantics(vectors: np.ndarray, control: str,
                                 seed: int = 2026) -> np.ndarray:
    """Build strict controls with identical tensor shape and adapter capacity."""
    vectors = np.asarray(vectors, dtype=np.float32)
    rng = np.random.default_rng(seed)
    if control == "correct":
        return vectors.copy()
    if control == "shuffle":
        return vectors[rng.permutation(vectors.shape[0])].copy()
    if control == "zero":
        return np.zeros_like(vectors)
    if control in {"random", "learned_id"}:
        random = rng.standard_normal(vectors.shape).astype(np.float32)
        random /= np.maximum(np.linalg.norm(random, axis=1, keepdims=True), 1e-6)
        return random
    raise ValueError(f"unknown channel semantic control: {control}")
