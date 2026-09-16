from types import SimpleNamespace

import pytest
import torch

from formal.config import DETERMINISTIC_MODELS, MODEL_NAMES
from layers.physical_semantic import N_TOKENS, TOKEN_WIDTH, resolve_physical_roles
from models.factory import build_model, registered_models

FEATURE_NAMES = [
    "lmd_totalirrad",
    "lmd_diffuseirrad",
    "lmd_temperature",
    "SolarZenithCos",
    "SolarDaySin",
    "SolarDayCos",
    "ClearSkyGHIProxy",
    "Target",
]
CLEAR_SKY_INDEX = FEATURE_NAMES.index("ClearSkyGHIProxy")


def _base_config():
    return SimpleNamespace(
        seq_len=96,
        pred_len=16,
        enc_in=len(FEATURE_NAMES),
        d_model=16,
        d_ff=32,
        n_heads=4,
        e_layers=1,
        dropout=0.0,
    )


def _config_for(name):
    config = _base_config()
    if name == "psrc":
        config.physical_semantic = {
            "roles": resolve_physical_roles(FEATURE_NAMES),
            "feature_mu": [0.0] * len(FEATURE_NAMES),
            "feature_sd": [1.0] * len(FEATURE_NAMES),
            "target_mu": 0.0,
            "target_sd": 1.0,
            "token_mean": [[0.0] * TOKEN_WIDTH for _ in range(N_TOKENS)],
            "token_scale": [[1.0] * TOKEN_WIDTH for _ in range(N_TOKENS)],
            "d_token": 16,
            "use_gate": True,
        }
    if name == "smart_persistence":
        config.solar_geometry = {
            "clear_sky_index": CLEAR_SKY_INDEX,
            "clear_future_pos": 0,
            "daylight_future_pos": None,
            "target_mu": 0.0,
            "target_sd": 1.0,
            "clear_mu": 0.5,
            "clear_sd": 0.4,
            "clear_sky_training_max": 1.0,
            "capacity": 1.0,
        }
    return config


@pytest.mark.parametrize("name", list(MODEL_NAMES))
def test_every_formal_model_uses_the_same_forward_contract(name):
    model = build_model(name, _config_for(name)).eval()
    x = torch.randn(1, 96, len(FEATURE_NAMES))
    cycle = torch.zeros(1, dtype=torch.long)
    future_solar = torch.randn(1, 16, 1)

    with torch.no_grad():
        prediction = model(x, cycle, future_solar, None, None)

    assert prediction.shape == (1, 16, 1)


@pytest.mark.parametrize("name", list(DETERMINISTIC_MODELS))
def test_parameter_free_baselines_have_no_trainable_tensors(name):
    model = build_model(name, _config_for(name))

    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_factory_registers_exactly_the_locked_model_set():
    assert set(registered_models()) == set(MODEL_NAMES)
    assert registered_models()[0] == "psrc"


def test_factory_rejects_unknown_model():
    with pytest.raises(KeyError, match="unknown model"):
        build_model("not_a_model", _base_config())


def test_smart_persistence_requires_geometry_config():
    with pytest.raises(ValueError, match="clear-sky geometry"):
        build_model("smart_persistence", _base_config())
