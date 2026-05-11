#!/usr/bin/env python3
"""Build ZCTA-year amenity features from category-level ZIP records."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "model-datasets" / "amenities"
DEMOGRAPHICS_FEATURES_PATH = PROJECT_ROOT / "model-ready" / "zcta_demographics_features.csv"
OUTPUT_DIR = PROJECT_ROOT / "model-ready"
CLEAN_OUTPUT = OUTPUT_DIR / "amenities_cleaned.csv"
FEATURE_OUTPUT = OUTPUT_DIR / "amenities_features.csv"
MERGED_LISTINGS_OUTPUT = OUTPUT_DIR / "listings_model_ready_with_amenities.csv"

EMPLOYEE_BUCKET_COLUMNS = [
    "emp_1_4",
    "emp_5_9",
    "emp_10_19",
    "emp_20_49",
    "emp_50_99",
    "emp_100_249",
    "emp_250_499",
    "emp_500_999",
    "emp_1000_plus",
]

AMENITY_NAME_MAP = {
    "Coffee Shops": "coffee_shops",
    "Fitness": "fitness",
    "Park": "parks",
    "Resturants": "restaurants",
    "Restaurants": "restaurants",
}


def normalize_amenity_name(value: str) -> str:
    if value in AMENITY_NAME_MAP:
        return AMENITY_NAME_MAP[value]
    return (
        str(value)
        .strip()
        .lower()
        .replace("&", "and")
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )


def load_amenities() -> pd.DataFrame:
    amenities = pd.read_csv(INPUT_PATH)
    amenities = amenities.rename(columns={"zip": "zcta5"})
    amenities["zcta5"] = amenities["zcta5"].astype(str).str.zfill(5)
    amenities["year"] = pd.to_numeric(amenities["year"], errors="raise").astype(int)
    amenities["amenity_type_raw"] = amenities["amenities"].astype(str).str.strip()
    amenities["amenity_type"] = amenities["amenity_type_raw"].map(normalize_amenity_name)

    numeric_columns = ["est", *EMPLOYEE_BUCKET_COLUMNS]
    for col in numeric_columns:
        amenities[col] = pd.to_numeric(amenities[col], errors="coerce").fillna(0)

    amenities["amenity_employee_bucket_total"] = amenities[EMPLOYEE_BUCKET_COLUMNS].sum(axis=1)
    amenities["amenity_est_bucket_gap"] = amenities["est"] - amenities["amenity_employee_bucket_total"]
    amenities["amenity_source_rows"] = 1

    clean_columns = [
        "zcta5",
        "year",
        "amenity_type_raw",
        "amenity_type",
        "est",
        *EMPLOYEE_BUCKET_COLUMNS,
        "amenity_employee_bucket_total",
        "amenity_est_bucket_gap",
        "amenity_source_rows",
    ]
    return amenities[clean_columns].sort_values(["zcta5", "year", "amenity_type"]).reset_index(drop=True)


def add_per_capita_features(features: pd.DataFrame) -> pd.DataFrame:
    if not DEMOGRAPHICS_FEATURES_PATH.exists():
        features["amenities_population_for_rate"] = np.nan
        return features

    demographics = pd.read_csv(
        DEMOGRAPHICS_FEATURES_PATH,
        usecols=["zcta5", "year", "total_population"],
        dtype={"zcta5": str},
    )
    demographics["zcta5"] = demographics["zcta5"].astype(str).str.zfill(5)
    demographics["year"] = pd.to_numeric(demographics["year"], errors="coerce").astype("Int64")
    features = features.merge(
        demographics.rename(columns={"total_population": "amenities_population_for_rate"}),
        on=["zcta5", "year"],
        how="left",
        validate="one_to_one",
    )

    denom = features["amenities_population_for_rate"].where(
        features["amenities_population_for_rate"].gt(0)
    )
    rate_columns = [c for c in features.columns if c.startswith("amenity_est_")]
    for col in rate_columns:
        rate_name = col.replace("amenity_est_", "amenity_per_10k_")
        features[rate_name] = features[col] / denom * 10_000

    return features


def build_features(clean: pd.DataFrame) -> pd.DataFrame:
    base = (
        clean.groupby(["zcta5", "year"], as_index=False)
        .agg(
            amenity_source_rows=("amenity_source_rows", "sum"),
            amenity_type_count=("amenity_type", "nunique"),
            amenity_est_total=("est", "sum"),
            amenity_employee_bucket_total=("amenity_employee_bucket_total", "sum"),
            amenity_est_bucket_gap=("amenity_est_bucket_gap", "sum"),
        )
        .sort_values(["zcta5", "year"])
    )
    base["amenities_observed"] = True

    est_wide = clean.pivot_table(
        index=["zcta5", "year"],
        columns="amenity_type",
        values="est",
        aggfunc="sum",
        fill_value=0,
    )
    est_wide.columns = [f"amenity_est_{col}" for col in est_wide.columns]
    est_wide = est_wide.reset_index()

    bucket_wide = clean.pivot_table(
        index=["zcta5", "year"],
        values=EMPLOYEE_BUCKET_COLUMNS,
        aggfunc="sum",
        fill_value=0,
    )
    bucket_wide.columns = [f"amenity_{col}" for col in bucket_wide.columns]
    bucket_wide = bucket_wide.reset_index()

    features = base.merge(est_wide, on=["zcta5", "year"], how="left", validate="one_to_one")
    features = features.merge(bucket_wide, on=["zcta5", "year"], how="left", validate="one_to_one")

    category_cols = sorted(c for c in features.columns if c.startswith("amenity_est_") and c != "amenity_est_total")
    bucket_cols = sorted(c for c in features.columns if c.startswith("amenity_emp_"))
    features[category_cols + bucket_cols] = features[category_cols + bucket_cols].fillna(0)

    features = add_per_capita_features(features)
    return features.sort_values(["zcta5", "year"]).reset_index(drop=True)


def write_merged_listings_feature(features: pd.DataFrame) -> None:
    listings_path = OUTPUT_DIR / "listings_model_ready.csv"
    if not listings_path.exists():
        return

    listings = pd.read_csv(listings_path, dtype={"zcta5": str})
    listings["zcta5"] = listings["zcta5"].astype(str).str.zfill(5)
    listings["year"] = pd.to_numeric(listings["year"], errors="raise").astype(int)

    merged = listings.merge(features, on=["zcta5", "year"], how="left", validate="many_to_one")
    merged["amenities_observed"] = merged["amenities_observed"].eq(True)
    merged.to_csv(MERGED_LISTINGS_OUTPUT, index=False)

    print(f"Wrote merged listings preview: {MERGED_LISTINGS_OUTPUT}")
    print(f"Rows: {len(merged):,}")
    print(f"Rows with amenities observed: {int(merged['amenities_observed'].sum()):,}")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    clean = load_amenities()
    features = build_features(clean)

    clean.to_csv(CLEAN_OUTPUT, index=False)
    features.to_csv(FEATURE_OUTPUT, index=False)

    print(f"Wrote cleaned amenities table: {CLEAN_OUTPUT}")
    print(f"Rows: {len(clean):,}")
    print(f"Years: {clean['year'].min()}-{clean['year'].max()}")
    print(f"ZCTAs: {clean['zcta5'].nunique():,}")
    print(f"Amenity types: {', '.join(sorted(clean['amenity_type'].unique()))}")
    print()
    print(f"Wrote amenities feature table: {FEATURE_OUTPUT}")
    print(f"Rows: {len(features):,}")
    print(f"Columns: {len(features.columns):,}")
    print(f"Rows missing population denominator: {int(features['amenities_population_for_rate'].isna().sum()):,}")
    print()
    write_merged_listings_feature(features)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
