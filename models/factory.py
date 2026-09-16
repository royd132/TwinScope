"""Local model factory with one shared forecasting interface."""

from importlib import import_module
from types import SimpleNamespace

_MODULES = {
    "psrc": "ours",
    "timemixer": "timemixer",
    "ampdnet": "ampdnet",
    "crossunet": "crossunet",
    "timesnet": "timesnet",
    "patchtst": "patchtst",
    "itransformer": "itransformer",
    "dlinear": "dlinear",
    "cyclenet": "cyclenet",
    "patchmlp": "patchmlp",
    "tcn": "tcn",
    "frets": "frets",
    "lstm": "lstm",
    "gru": "gru",
    "timexer": "timexer",
    "persistence": "persistence",
    "seasonal_naive": "seasonal_naive",
    "smart_persistence": "smart_persistence",
}

_DEFAULTS = {
    "task_name": "long_term_forecast",
    "features": "MS",
    "label_len": 16,
    "cycle": 96,
    "model_type": "mlp",
    "use_revin": 1,
    "useweather": False,
    "use_norm": True,
    "output_attention": False,
    "patch_len_arryay": [48, 24, 12, 6],
    "e_layers": 3,
    "down_sampling_window": 2,
    "down_sampling_layers": 3,
    "down_sampling_method": "avg",
    "channel_independence": 0,
    "moving_avg": 25,
    "use_future_temporal_feature": 0,
    "d_ff": 256,
    "dropout": 0.1,
    "embed": "fixed",
    "freq": "h",
    "top_k": 3,
    "num_kernels": 4,
    "decomp_method": "moving_avg",
    "seg_len": 12,
    "n_heads": 4,
    "factor": 10,
    "usenonlinearproject": True,
    "usebottle": True,
    "convmerge": False,
    "swichchannel": False,
    "twofilter": True,
    "c_out": 1,
}


def build_model(name, configs):
    key = str(name).lower().replace("-", "_")
    if key not in _MODULES:
        raise KeyError(f"unknown model: {name}; choose from {sorted(_MODULES)}")
    values = dict(_DEFAULTS)
    values.update(vars(configs))
    values["label_len"] = values.get("pred_len", values["label_len"])
    values["d_ff"] = max(values["d_ff"], 2 * values.get("d_model", 128))
    return import_module(f".{_MODULES[key]}", __package__).Model(
        SimpleNamespace(**values)
    )


def registered_models():
    """Return formal model names in stable report order."""
    return tuple(_MODULES)
