import numpy as np
import pytest

from utils.prompt_features import _window_tokens, build_or_load_prompt_features

FEATURE_NAMES = [
    "Temperature",
    "Humidity",
    "GHI",
    "DHI",
    "ClockSin",
    "ClockCos",
    "SolarGeometryProxy",
    "DaylightHeuristic",
    "Target",
]


def prompt_stats():
    return {
        "dt_hours": 0.25,
        "seq_len": 96,
        "feature_names": FEATURE_NAMES,
        "feature_mu": [0.0] * len(FEATURE_NAMES),
        "feature_sd": [1.0] * len(FEATURE_NAMES),
        "quality_indices": [],
    }


def test_unknown_offline_encoder_is_rejected_before_cache_or_model_access(tmp_path):
    x = np.zeros((145, len(FEATURE_NAMES)), dtype=np.float32)
    with pytest.raises(ValueError, match="unsupported prompt encoder"):
        build_or_load_prompt_features(
            x,
            [0],
            prompt_stats(),
            48,
            tmp_path / "prompt.npz",
            tmp_path / "missing_hf_cache",
            prompt_version=10,
            include_future_solar=False,
            encoder_kind="misspelled",
        )


def test_full_history_prompt_is_order_sensitive_and_history_only():
    x = np.zeros((96, len(FEATURE_NAMES)), dtype=np.float32)
    x[:, -1] = np.linspace(-1.0, 1.0, 96, dtype=np.float32)
    texts, numeric = _window_tokens(x, prompt_stats(), 48, None, 10)

    changed = x.copy()
    changed[[10, 11], -1] = changed[[11, 10], -1]
    changed_texts, changed_numeric = _window_tokens(
        changed,
        prompt_stats(),
        48,
        np.full(48, 999.0, dtype=np.float32),
        10,
    )

    assert len(texts) == len(FEATURE_NAMES)
    assert numeric.shape == (len(FEATURE_NAMES), 104)
    assert texts[-1] != changed_texts[-1]
    assert not np.array_equal(numeric[-1], changed_numeric[-1])

    same_texts, same_numeric = _window_tokens(
        x,
        prompt_stats(),
        48,
        np.full(48, -999.0, dtype=np.float32),
        10,
    )
    assert texts == same_texts
    np.testing.assert_array_equal(numeric, same_numeric)
