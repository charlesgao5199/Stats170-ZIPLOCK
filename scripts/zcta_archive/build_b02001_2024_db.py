#!/usr/bin/env python3
"""Build a SQLite database for ACS 2024 B02001 data for selected ZCTAs."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "zcta_archive" / "raw"

INPUT_CSV = RAW_DIR / "acs_2024_demographics_ca_zcta.csv"
OUTPUT_DB = RAW_DIR / "acs_2024_b02001_selected_zctas.db"
TABLE_NAME = "b02001_2024_selected_zctas"

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


def build_db(input_csv: Path, output_db: Path, table_name: str, zips: list[str]) -> None:
    usecols = ["ZCTA5", "NAME", "GEO_ID"]
    header_df = pd.read_csv(input_csv, nrows=0)
    b02001_cols = [c for c in header_df.columns if c.startswith("B02001")]
    if not b02001_cols:
        raise ValueError("No B02001 columns found in input CSV.")
    usecols.extend(b02001_cols)

    df = pd.read_csv(input_csv, usecols=usecols, dtype=str, low_memory=False)
    df["ZCTA5"] = df["ZCTA5"].astype(str).str.strip().str.zfill(5)

    normalized_zips = [str(z).zfill(5) for z in zips]
    df = df[df["ZCTA5"].isin(set(normalized_zips))].copy()
    if df.empty:
        raise RuntimeError("No matching ZIP/ZCTA rows found for requested list.")

    zip_order = {z: i for i, z in enumerate(normalized_zips)}
    df["__zip_order"] = df["ZCTA5"].map(zip_order)
    df = df.sort_values("__zip_order").drop(columns="__zip_order")

    output_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(output_db) as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False)

    print(f"Wrote DB: {output_db}")
    print(f"Table: {table_name}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build SQLite DB for ACS 2024 B02001 for selected ZCTAs."
    )
    parser.add_argument("--input", default=str(INPUT_CSV), help="Input ACS CSV path.")
    parser.add_argument("--output-db", default=str(OUTPUT_DB), help="Output SQLite DB path.")
    parser.add_argument("--table", default=TABLE_NAME, help="SQLite table name.")
    parser.add_argument(
        "--zips",
        nargs="+",
        default=DEFAULT_ZIPS,
        help="ZIP/ZCTA codes to include.",
    )
    args = parser.parse_args()

    build_db(
        input_csv=Path(args.input),
        output_db=Path(args.output_db),
        table_name=args.table,
        zips=args.zips,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
