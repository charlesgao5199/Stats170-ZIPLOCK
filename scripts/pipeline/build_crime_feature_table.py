#!/usr/bin/env python3
"""Build ZCTA-year crime features from county-estimated ZIP records."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "model-datasets" / "crimes_estimates"
OUTPUT_DIR = PROJECT_ROOT / "model-ready"
CLEAN_OUTPUT = OUTPUT_DIR / "crime_estimates_cleaned.csv"
FEATURE_OUTPUT = OUTPUT_DIR / "crime_features.csv"
MERGED_LISTINGS_OUTPUT = OUTPUT_DIR / "listings_model_ready_with_crime.csv"

COUNT_COLUMNS = [
    "est_Violent_Crime_Total",
    "est_Homicide_Total",
    "est_Rape_Total",
    "est_Robbery_sum",
    "est_Aggravated_Assaults_Total",
    "est_Property_Crime_Total",
    "est_Burglary_Total",
    "est_Vehicle_Theft_Crime_Total",
    "est_Larceny_Thefts_Total",
]

RAW_COUNTY_TOTAL_COLUMNS = [
    "Violent_Crime_Total",
    "Homicide_Total",
    "Rape_Total",
    "Robbery_sum",
    "Aggravated_Assaults_Total",
    "Property_Crime_Total",
    "Burglary_Total",
    "Vehicle_Theft_Crime_Total",
    "Larceny_Thefts_Total",
]

FEATURE_NAME_MAP = {
    "est_Violent_Crime_Total": "crime_violent_est",
    "est_Homicide_Total": "crime_homicide_est",
    "est_Rape_Total": "crime_rape_est",
    "est_Robbery_sum": "crime_robbery_est",
    "est_Aggravated_Assaults_Total": "crime_aggravated_assault_est",
    "est_Property_Crime_Total": "crime_property_est",
    "est_Burglary_Total": "crime_burglary_est",
    "est_Vehicle_Theft_Crime_Total": "crime_vehicle_theft_est",
    "est_Larceny_Thefts_Total": "crime_larceny_theft_est",
}


def normalize_county_name(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\s+County$", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
    )


def load_crime() -> pd.DataFrame:
    crime = pd.read_csv(INPUT_PATH)
    crime = crime.rename(
        columns={
            "Zipcode": "zcta5",
            "County": "county",
            "Year": "year",
            "Population": "county_population_source",
            "total_population": "crime_population_for_rate",
            "Ratio": "crime_allocation_ratio",
            "Total_Crime": "source_total_crime_double_counted",
        }
    )
    crime["zcta5"] = crime["zcta5"].astype(str).str.zfill(5)
    crime["county"] = normalize_county_name(crime["county"])
    crime["year"] = pd.to_numeric(crime["year"], errors="raise").astype(int)

    numeric_columns = [
        "county_population_source",
        "crime_population_for_rate",
        "crime_allocation_ratio",
        "source_total_crime_double_counted",
        *COUNT_COLUMNS,
        *RAW_COUNTY_TOTAL_COLUMNS,
    ]
    for col in numeric_columns:
        crime[col] = pd.to_numeric(crime[col], errors="coerce")

    crime["crime_index_est"] = (
        crime["est_Violent_Crime_Total"] + crime["est_Property_Crime_Total"]
    )
    crime["violent_component_rounding_gap"] = (
        crime["est_Homicide_Total"]
        + crime["est_Rape_Total"]
        + crime["est_Robbery_sum"]
        + crime["est_Aggravated_Assaults_Total"]
        - crime["est_Violent_Crime_Total"]
    )
    crime["property_component_rounding_gap"] = (
        crime["est_Burglary_Total"]
        + crime["est_Vehicle_Theft_Crime_Total"]
        + crime["est_Larceny_Thefts_Total"]
        - crime["est_Property_Crime_Total"]
    )
    crime["crime_source_rows"] = 1

    clean_columns = [
        "zcta5",
        "year",
        "county",
        "county_population_source",
        "crime_population_for_rate",
        "crime_allocation_ratio",
        *COUNT_COLUMNS,
        "crime_index_est",
        "source_total_crime_double_counted",
        "violent_component_rounding_gap",
        "property_component_rounding_gap",
        "crime_source_rows",
    ]
    return crime[clean_columns].sort_values(["zcta5", "year", "county"]).reset_index(drop=True)


def build_features(clean: pd.DataFrame) -> pd.DataFrame:
    agg_spec = {
        "crime_source_rows": ("crime_source_rows", "sum"),
        "crime_county_count": ("county", "nunique"),
        "crime_counties": ("county", lambda values: "; ".join(sorted(values.unique()))),
        "crime_population_for_rate": ("crime_population_for_rate", "max"),
        "crime_allocation_ratio_sum": ("crime_allocation_ratio", "sum"),
        "crime_source_total_double_counted_sum": ("source_total_crime_double_counted", "sum"),
        "violent_component_rounding_gap": ("violent_component_rounding_gap", "sum"),
        "property_component_rounding_gap": ("property_component_rounding_gap", "sum"),
        "crime_index_est": ("crime_index_est", "sum"),
    }
    for col in COUNT_COLUMNS:
        agg_spec[FEATURE_NAME_MAP[col]] = (col, "sum")

    features = clean.groupby(["zcta5", "year"], as_index=False).agg(**agg_spec)
    features["crime_observed"] = True
    features["crime_multi_county_zcta"] = features["crime_county_count"].gt(1)

    denom = features["crime_population_for_rate"].where(
        features["crime_population_for_rate"].gt(0)
    )
    count_feature_cols = ["crime_index_est", *FEATURE_NAME_MAP.values()]
    for col in count_feature_cols:
        features[f"{col}_per_1k"] = features[col] / denom * 1_000
        features[f"log1p_{col}"] = np.log1p(features[col])

    features["crime_has_zero_or_missing_population"] = denom.isna()
    return features.sort_values(["zcta5", "year"]).reset_index(drop=True)


def write_merged_listings_feature(features: pd.DataFrame) -> None:
    listings_path = OUTPUT_DIR / "listings_model_ready.csv"
    if not listings_path.exists():
        return

    listings = pd.read_csv(listings_path, dtype={"zcta5": str})
    listings["zcta5"] = listings["zcta5"].astype(str).str.zfill(5)
    listings["year"] = pd.to_numeric(listings["year"], errors="raise").astype(int)

    merged = listings.merge(features, on=["zcta5", "year"], how="left", validate="many_to_one")
    merged["crime_observed"] = merged["crime_observed"].eq(True)
    merged.to_csv(MERGED_LISTINGS_OUTPUT, index=False)

    print(f"Wrote merged listings preview: {MERGED_LISTINGS_OUTPUT}")
    print(f"Rows: {len(merged):,}")
    print(f"Rows with crime observed: {int(merged['crime_observed'].sum()):,}")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    clean = load_crime()
    features = build_features(clean)

    clean.to_csv(CLEAN_OUTPUT, index=False)
    features.to_csv(FEATURE_OUTPUT, index=False)

    print(f"Wrote cleaned crime estimates table: {CLEAN_OUTPUT}")
    print(f"Rows: {len(clean):,}")
    print(f"Years: {clean['year'].min()}-{clean['year'].max()}")
    print(f"ZCTAs: {clean['zcta5'].nunique():,}")
    print()
    print(f"Wrote crime feature table: {FEATURE_OUTPUT}")
    print(f"Rows: {len(features):,}")
    print(f"Columns: {len(features.columns):,}")
    print(f"Multi-county ZCTA-year rows: {int(features['crime_multi_county_zcta'].sum()):,}")
    print(
        "Rows with zero/missing population denominator: "
        f"{int(features['crime_has_zero_or_missing_population'].sum()):,}"
    )
    print()
    write_merged_listings_feature(features)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
