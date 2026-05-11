#!/usr/bin/env python3
"""Build cleaned and regime-aware school quality tables for modeling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "model-datasets" / "school_quality_updated"
DEMOGRAPHICS_FEATURES_PATH = PROJECT_ROOT / "model-ready" / "zcta_demographics_features.csv"
OUTPUT_DIR = PROJECT_ROOT / "model-ready"
CLEAN_OUTPUT = OUTPUT_DIR / "school_quality_cleaned.csv"
FEATURE_OUTPUT = OUTPUT_DIR / "school_quality_features.csv"

NUMERIC_COLUMNS = [
    "n_schools",
    "avg_pct_met_ela",
    "avg_pct_met_math",
    "avg_pct_met_overall",
    "median_pct_met_overall",
    "min_pct_met_overall",
    "max_pct_met_overall",
    "total_students_tested",
]

ZSCORE_COLUMNS = [
    "avg_pct_met_ela",
    "avg_pct_met_math",
    "avg_pct_met_overall",
    "median_pct_met_overall",
]


def zscore_within_group(series: pd.Series) -> pd.Series:
    """Compute a z-score within a group, returning 0 when variation is zero."""
    series = pd.to_numeric(series, errors="coerce")
    mean = series.mean()
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.where(series.notna(), 0.0, np.nan), index=series.index)
    return (series - mean) / std


def percentile_within_group(series: pd.Series) -> pd.Series:
    """Return percentile rank in [0, 1] within each assessment-year group."""
    return series.rank(pct=True, method="average")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    school = pd.read_csv(INPUT_PATH, dtype=str, low_memory=False)
    school["zcta5"] = school["zcta5"].astype(str).str.zfill(5)
    school["year"] = pd.to_numeric(school["year"], errors="coerce").astype("Int64")

    for col in NUMERIC_COLUMNS:
        school[col] = pd.to_numeric(school[col], errors="coerce")

    # Keep only zcta-year keys that survive the demographics cleaning step.
    valid_demo_keys = pd.read_csv(
        DEMOGRAPHICS_FEATURES_PATH,
        usecols=["zcta5", "year"],
        dtype={"zcta5": str, "year": "Int64"},
    ).drop_duplicates()
    valid_demo_keys["zcta5"] = valid_demo_keys["zcta5"].astype(str).str.zfill(5)

    school = school.merge(
        valid_demo_keys.assign(valid_demographics_key=True),
        on=["zcta5", "year"],
        how="inner",
    )

    school["assessment_type"] = school["assessment_type"].astype(str)
    school["is_caaspp"] = school["assessment_type"].eq("CAASPP")
    school["is_cst"] = school["assessment_type"].eq("CST")
    school["school_year_complete"] = ~school["year"].isin([2014, 2020])

    school["total_students_tested_reported_positive"] = school["total_students_tested"].gt(0)
    school["total_students_tested_unreliable"] = school["total_students_tested"].fillna(0).eq(0)
    school["overall_score_range"] = school["max_pct_met_overall"] - school["min_pct_met_overall"]
    school["school_quality_extended_eligible"] = school["avg_pct_met_overall"].notna()

    clean_columns = [
        "zcta5",
        "year",
        "assessment_type",
        "is_caaspp",
        "is_cst",
        "school_year_complete",
        "n_schools",
        "avg_pct_met_ela",
        "avg_pct_met_math",
        "avg_pct_met_overall",
        "median_pct_met_overall",
        "min_pct_met_overall",
        "max_pct_met_overall",
        "overall_score_range",
        "total_students_tested",
        "total_students_tested_reported_positive",
        "total_students_tested_unreliable",
        "school_quality_extended_eligible",
    ]
    clean = school[clean_columns].sort_values(["zcta5", "year"]).reset_index(drop=True)
    clean.to_csv(CLEAN_OUTPUT, index=False)

    feature = clean.copy()
    feature["log1p_n_schools"] = np.log1p(feature["n_schools"])

    grouped = feature.groupby(["assessment_type", "year"], dropna=False)
    for col in ZSCORE_COLUMNS:
        feature[f"{col}_z_assessment_year"] = grouped[col].transform(zscore_within_group)
        feature[f"{col}_pct_assessment_year"] = grouped[col].transform(percentile_within_group)

    feature_columns = [
        "zcta5",
        "year",
        "assessment_type",
        "is_caaspp",
        "is_cst",
        "school_year_complete",
        "school_quality_extended_eligible",
        "n_schools",
        "log1p_n_schools",
        "avg_pct_met_ela",
        "avg_pct_met_math",
        "avg_pct_met_overall",
        "median_pct_met_overall",
        "min_pct_met_overall",
        "max_pct_met_overall",
        "overall_score_range",
        "total_students_tested_reported_positive",
        "total_students_tested_unreliable",
        "avg_pct_met_ela_z_assessment_year",
        "avg_pct_met_math_z_assessment_year",
        "avg_pct_met_overall_z_assessment_year",
        "median_pct_met_overall_z_assessment_year",
        "avg_pct_met_ela_pct_assessment_year",
        "avg_pct_met_math_pct_assessment_year",
        "avg_pct_met_overall_pct_assessment_year",
        "median_pct_met_overall_pct_assessment_year",
    ]
    feature = feature[feature_columns].sort_values(["zcta5", "year"]).reset_index(drop=True)
    feature.to_csv(FEATURE_OUTPUT, index=False)

    print(f"Wrote cleaned school quality table: {CLEAN_OUTPUT}")
    print(f"Rows: {len(clean):,}")
    print(f"Columns: {len(clean.columns):,}")
    print()
    print(f"Wrote school quality feature table: {FEATURE_OUTPUT}")
    print(f"Rows: {len(feature):,}")
    print(f"Columns: {len(feature.columns):,}")
    print(f"Rows with CAASPP: {int(feature['is_caaspp'].sum()):,}")
    print(f"Rows with CST: {int(feature['is_cst'].sum()):,}")
    print(
        "Rows with positive reported tested counts: "
        f"{int(feature['total_students_tested_reported_positive'].sum()):,}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
