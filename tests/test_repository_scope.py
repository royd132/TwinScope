from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_repository_contains_only_formal_dataset_files():
    names = {path.name for path in (ROOT / "data").glob("*.csv")}
    assert names == {
        "91-Site_DKA-M9_B-Phase.csv",
        "98-Site_DKA-M8_B-Phase.csv",
        "DKASC_site1a_15min.csv",
        "DKASC_site19_15min.csv",
        "DKASC_site31_15min.csv",
        "DKASC_site9a_15min.csv",
        "HKUST_UG_Hall4_15min_causal_physics.csv",
        "PVOD_station00_15min.csv",
        "PVOD_station01_15min.csv",
        "PVOD_station02_15min.csv",
        "PVOD_station03_15min.csv",
    }


def test_legacy_experiment_trees_are_removed():
    for name in ("datasets", "experiments", "external", "results"):
        assert not (ROOT / name).exists()


def test_model_and_layer_surface_is_locked():
    model_names = {path.name for path in (ROOT / "models").glob("*.py")}
    assert model_names == {
        "__init__.py",
        "ampdnet.py",
        "crossunet.py",
        "cyclenet.py",
        "dlinear.py",
        "factory.py",
        "itransformer.py",
        "ours.py",
        "patchmlp.py",
        "patchtst.py",
        "persistence.py",
        "seasonal_naive.py",
        "smart_persistence.py",
        "tcn.py",
        "frets.py",
        "lstm.py",
        "gru.py",
        "timexer.py",
        "timemixer.py",
        "timesnet.py",
    }
    layer_names = {path.name for path in (ROOT / "layers").glob("*.py")}
    assert layer_names == {
        "__init__.py",
        "AMPD_Embed.py",
        "AMPD_SelfAttention_Family.py",
        "AMPD_Transformer_EncDec.py",
        "Autoformer_EncDec_timemixer.py",
        "Conv_Blocks.py",
        "CrossUentattn.py",
        "CrossUnet_EncDec.py",
        "Embed.py",
        "Embed_patchmlp.py",
        "Embed_timemixer.py",
        "StandardNorm_timemixer.py",
        "corpatch.py",
        "pc_fra.py",
        "physical_semantic.py",
        "revin.py",
        "residual_corrector.py",
    }


def test_only_required_utility_module_remains():
    utility_names = {path.name for path in (ROOT / "utils").glob("*.py")}
    assert utility_names == {"__init__.py", "masking.py"}


def test_exploratory_tools_are_removed():
    assert not (ROOT / "tools").exists()
    assert not (ROOT / "tools_bench_amp.py").exists()
    assert not (ROOT / "formal" / "lr_fix_pvod00.py").exists()
