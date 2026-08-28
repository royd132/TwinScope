"""Prepare the TFCNet paper's DKASC Site 1B/24 data at 15-minute resolution.

The author repository publishes already-imputed 5-minute CSV files covering
March-May 2021.  The files include one extra row at exactly 2021-06-01 00:00,
while the paper reports 26,496 points per site.  We therefore use the explicit
half-open interval [2021-03-01 00:00, 2021-06-01 00:00), then average each
three consecutive instantaneous measurements into one 15-minute record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "datasets" / "australia" / "DKASC_paper_2021"
RAW_ROOT = DATA_ROOT / "raw"
START = pd.Timestamp("2021-03-01 00:00:00")
END = pd.Timestamp("2021-06-01 00:00:00")
SOURCE_BASE = (
    "https://raw.githubusercontent.com/laowu-code/"
    "TimeFrequency_EndogenousExogenousDecoupling_PVPowerForecast/main"
)

SITES = {
    "site1b": {
        "source_name": "site_1B_2021_03_01_2021_06_01_5min.csv",
        "source_url": f"{SOURCE_BASE}/data/site_1B/site_1B_2021_3_2021_5_5min.csv",
        "source_sha256": "8386e5d0d02ce4c4e01de02c510c05112dca4ef980b84c235b854d0b2da5372c",
        "output_name": "DKASC_site1B_2021MarMay_15min.csv",
        "paper_dataset": "A",
        "rated_capacity_kw": 23.4,
    },
    "site24": {
        "source_name": "site_24_2021_03_01_2021_06_01_5min.csv",
        "source_url": f"{SOURCE_BASE}/data/site_24/site_24_2021_3_2021_5_5min.csv",
        "source_sha256": "6da7cc6643c2dab8a234d8b059ce31bd73c1bc933050dd95c10af1d13b84cd04",
        "output_name": "DKASC_site24_2021MarMay_15min.csv",
        "paper_dataset": "B",
        "rated_capacity_kw": 6.1,
    },
}

SOURCE_COLUMNS = [
    "timestamp",
    "Active_Power",
    "Weather_Temperature_Celsius",
    "Weather_Relative_Humidity",
    "Global_Horizontal_Radiation",
    "Diffuse_Horizontal_Radiation",
]
OUTPUT_COLUMNS = [
    "date",
    "Target",
    "Weather_Temperature_Celsius",
    "Weather_Relative_Humidity",
    "Global_Horizontal_Radiation",
    "Diffuse_Horizontal_Radiation",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "Codex-DKASC-reproduction"})
    with urlopen(request, timeout=120) as response, destination.open("wb") as handle:
        while block := response.read(1024 * 1024):
            handle.write(block)


def prepare_site(site: str, spec: dict, allow_download: bool) -> dict:
    source = RAW_ROOT / spec["source_name"]
    output = DATA_ROOT / spec["output_name"]
    if not source.exists():
        if not allow_download:
            raise FileNotFoundError(f"missing source file: {source}; rerun with --download")
        download(spec["source_url"], source)

    actual_source_hash = sha256(source)
    if actual_source_hash != spec["source_sha256"]:
        raise ValueError(
            f"source hash mismatch for {site}: expected {spec['source_sha256']}, "
            f"got {actual_source_hash}"
        )

    frame = pd.read_csv(source)
    if list(frame.columns) != SOURCE_COLUMNS:
        raise ValueError(f"unexpected columns in {source}: {list(frame.columns)}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    if frame["timestamp"].duplicated().any():
        raise ValueError(f"duplicate timestamps in {source}")
    frame = frame.sort_values("timestamp").reset_index(drop=True)

    paper_interval = frame.loc[
        (frame["timestamp"] >= START) & (frame["timestamp"] < END)
    ].copy()
    if len(paper_interval) != 26_496:
        raise ValueError(
            f"{site} must contain 26,496 paper-interval rows, got {len(paper_interval)}"
        )
    expected_5min = pd.date_range(START, END, freq="5min", inclusive="left")
    if not paper_interval["timestamp"].reset_index(drop=True).equals(
        pd.Series(expected_5min, name="timestamp")
    ):
        raise ValueError(f"{site} is not a complete regular 5-minute sequence")
    if paper_interval[SOURCE_COLUMNS[1:]].isna().any().any():
        raise ValueError(f"{site} still contains missing values after author preprocessing")

    indexed = paper_interval.set_index("timestamp")
    group_sizes = indexed["Active_Power"].resample(
        "15min", label="left", closed="left"
    ).size()
    if not (group_sizes == 3).all():
        raise ValueError(f"{site} contains incomplete 15-minute bins")
    prepared = indexed.resample(
        "15min", label="left", closed="left"
    ).mean(numeric_only=True)
    prepared = prepared.rename(columns={"Active_Power": "Target"})
    prepared.index.name = "date"
    prepared = prepared.reset_index()[OUTPUT_COLUMNS]

    if len(prepared) != 8_832:
        raise ValueError(f"{site} must contain 8,832 15-minute rows, got {len(prepared)}")
    if prepared.isna().any().any():
        raise ValueError(f"{site} prepared data contains missing values")
    output.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output, index=False, date_format="%Y-%m-%d %H:%M:%S")

    return {
        "site": site,
        "paper_dataset": spec["paper_dataset"],
        "rated_capacity_kw": spec["rated_capacity_kw"],
        "source_url": spec["source_url"],
        "source_path": str(source.relative_to(ROOT)),
        "source_sha256": actual_source_hash,
        "source_rows_including_endpoint": int(len(frame)),
        "source_rows_paper_interval": int(len(paper_interval)),
        "source_interval": {"start_inclusive": str(START), "end_exclusive": str(END)},
        "author_missing_value_policy": "monthly mean; source CSV is already complete",
        "resampling": {
            "source_minutes": 5,
            "target_minutes": 15,
            "aggregation": "arithmetic mean of three instantaneous samples",
            "label": "left",
            "closed": "left",
        },
        "output_path": str(output.relative_to(ROOT)),
        "output_sha256": sha256(output),
        "output_rows": int(len(prepared)),
        "output_start": str(prepared["date"].iloc[0]),
        "output_end": str(prepared["date"].iloc[-1]),
        "target_min_kw": float(prepared["Target"].min()),
        "target_max_kw": float(prepared["Target"].max()),
        "target_mean_kw": float(prepared["Target"].mean()),
        "columns": list(prepared.columns),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download",
        action="store_true",
        help="download missing author-repository CSVs before preparing",
    )
    args = parser.parse_args()
    records = [prepare_site(site, spec, args.download) for site, spec in SITES.items()]
    manifest = {
        "dataset_family": "DKASC Alice Springs",
        "paper_doi": "10.1016/j.energy.2025.139481",
        "paper_author_repository": (
            "https://github.com/laowu-code/"
            "TimeFrequency_EndogenousExogenousDecoupling_PVPowerForecast"
        ),
        "important_difference_from_paper": (
            "The paper evaluates these sites at 5-minute resolution; this project "
            "uses a user-requested 15-minute mean aggregation."
        ),
        "sites": records,
    }
    manifest_path = DATA_ROOT / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
