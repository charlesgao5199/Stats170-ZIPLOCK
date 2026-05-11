#!/usr/bin/env python3
"""Plot B01003_001E time series for selected California ZCTAs."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "outputs" / "cache"
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR / "xdg"))
(CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
(CACHE_DIR / "xdg").mkdir(parents=True, exist_ok=True)

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import StrMethodFormatter  # noqa: E402


ZIP_COL = "ZCTA5"
VALUE_COL = "B01003_001E"
FILE_PATTERN = "acs_*_demographics_ca_zcta.csv"
YEAR_RE = re.compile(r"acs_(\d{4})_demographics_ca_zcta\.csv$")

DEFAULT_ZIPS = [
    "92602",
    "92603",
    "92604",
    "92606",
    "92612",
    "92614",
    "92617",
    "92618",
    "92620",
    "92697",
]


def load_series_rows(data_dir: Path, target_zips: list[str]) -> pd.DataFrame:
    records = []
    target_set = set(target_zips)

    for path in sorted(data_dir.glob(FILE_PATTERN)):
        match = YEAR_RE.match(path.name)
        if not match:
            continue
        year = int(match.group(1))

        try:
            df = pd.read_csv(path, usecols=[ZIP_COL, VALUE_COL], dtype=str)
        except ValueError as exc:
            raise ValueError(
                f"{path.name} is missing required columns {ZIP_COL}/{VALUE_COL}"
            ) from exc

        df[ZIP_COL] = df[ZIP_COL].astype(str).str.strip().str.zfill(5)
        df = df[df[ZIP_COL].isin(target_set)].copy()
        if df.empty:
            continue

        df[VALUE_COL] = pd.to_numeric(
            df[VALUE_COL].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )
        df = df.drop_duplicates(subset=[ZIP_COL], keep="first")
        df["year"] = year
        records.append(df[[ZIP_COL, "year", VALUE_COL]])

    if not records:
        raise RuntimeError("No matching data found for target ZIP codes.")

    result = pd.concat(records, ignore_index=True)
    return result.sort_values(["year", ZIP_COL]).reset_index(drop=True)


def plot_series(df: pd.DataFrame, target_zips: list[str], out_path: Path) -> None:
    pivot = df.pivot(index="year", columns=ZIP_COL, values=VALUE_COL)
    pivot = pivot.reindex(columns=target_zips)
    pivot = pivot.sort_index()

    ax = pivot.plot(figsize=(12, 7), marker="o", linewidth=2)
    ax.set_title("ACS B01003_001E Over Time for Selected CA ZCTAs")
    ax.set_xlabel("Year")
    ax.set_ylabel("Population (Estimate)")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.4)
    ax.legend(title="ZCTA5", ncol=2, frameon=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot Total Population time series for ZCTAs in Irvine."
    )
    parser.add_argument(
        "--data-dir",
        default=str(PROJECT_ROOT / "data" / "zcta_archive" / "raw"),
        help="Directory containing acs_YYYY_demographics_ca_zcta.csv files.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs" / "zcta-exploration" / "b01003_selected_zctas_timeseries.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--zips",
        nargs="+",
        default=DEFAULT_ZIPS,
        help="Space-separated ZCTA5 codes to plot.",
    )
    args = parser.parse_args()

    target_zips = [str(z).zfill(5) for z in args.zips]
    df = load_series_rows(Path(args.data_dir), target_zips)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plot_series(df, target_zips, output)

    years = sorted(df["year"].unique().tolist())
    print(f"Loaded years: {years[0]}-{years[-1]} ({len(years)} years)")
    print(f"ZIP codes requested: {', '.join(target_zips)}")
    print(f"Output plot: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
