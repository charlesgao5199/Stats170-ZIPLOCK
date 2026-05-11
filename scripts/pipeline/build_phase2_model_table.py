#!/usr/bin/env python3
"""Merge phase-2 feature tables into one listings-keyed modeling table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_READY_DIR = PROJECT_ROOT / "model-ready"
LISTINGS_PATH = MODEL_READY_DIR / "listings_model_ready.csv"
DEMOGRAPHICS_PATH = MODEL_READY_DIR / "zcta_demographics_features.csv"
SCHOOL_PATH = MODEL_READY_DIR / "school_quality_features.csv"
MINIMUM_WAGE_PATH = MODEL_READY_DIR / "zcta_minimum_wage_features.csv"
CRIME_PATH = MODEL_READY_DIR / "crime_features.csv"
AMENITIES_PATH = MODEL_READY_DIR / "amenities_features.csv"
OUTPUT_PATH = MODEL_READY_DIR / "phase2_model_ready.csv"


def read_keyed_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"zcta5": str}, low_memory=False)
    frame["zcta5"] = frame["zcta5"].astype(str).str.zfill(5)
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
    return frame


def merge_features(base: pd.DataFrame, path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        base[f"phase2_has_{label}"] = False
        return base

    features = read_keyed_csv(path)
    before_cols = set(base.columns)
    merged = base.merge(features, on=["zcta5", "year"], how="left", validate="many_to_one")
    new_cols = [col for col in merged.columns if col not in before_cols]
    merged[f"phase2_has_{label}"] = (
        merged[new_cols].notna().any(axis=1) if new_cols else False
    )
    return merged


def main() -> int:
    listings = read_keyed_csv(LISTINGS_PATH)
    model = listings.copy()

    for path, label in [
        (DEMOGRAPHICS_PATH, "demographics"),
        (SCHOOL_PATH, "school_quality"),
        (MINIMUM_WAGE_PATH, "minimum_wage"),
        (CRIME_PATH, "crime"),
        (AMENITIES_PATH, "amenities"),
    ]:
        model = merge_features(model, path, label)

    model.to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote phase-2 model table: {OUTPUT_PATH}")
    print(f"Rows: {len(model):,}")
    print(f"Columns: {len(model.columns):,}")
    for label in [
        "demographics",
        "school_quality",
        "minimum_wage",
        "crime",
        "amenities",
    ]:
        col = f"phase2_has_{label}"
        print(f"Rows with {label}: {int(model[col].sum()):,}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
