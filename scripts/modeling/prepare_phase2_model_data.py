#!/usr/bin/env python3
"""Prepare leakage-safe train/test data for phase-2 housing price models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "model-ready" / "phase2_model_ready.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase2-models" / "model-data"

TRAIN_START_YEAR = 2012
TRAIN_END_YEAR = 2022
TEST_YEAR = 2023
TARGET_COLUMN = "next_year_annual_median_sale_price"
TARGET_ALIAS = "y_next_year_median_sale_price"
LOG_TARGET_ALIAS = "y_log_next_year_median_sale_price"

IDENTIFIER_COLUMNS = {
    "zcta5",
    "geo_id",
    "zcta_name",
}

EXPLICIT_LEAKAGE_COLUMNS = {
    TARGET_COLUMN,
    "next_year_months_observed",
    "next_year_full_year_coverage",
    "target_median_sale_price_pct_change_next_year",
    "target_median_sale_price_log_change_next_year",
    "has_next_year_target",
    "target_next_year_complete",
    "baseline_model_eligible",
}

OUTPUT_FILES = {
    "train": OUTPUT_DIR / "phase2_train_2012_2022.csv",
    "test": OUTPUT_DIR / "phase2_test_2023.csv",
    "feature_schema": OUTPUT_DIR / "phase2_feature_schema.csv",
    "split_summary": OUTPUT_DIR / "phase2_split_summary.csv",
    "cv_folds": OUTPUT_DIR / "phase2_time_blocked_cv_folds.csv",
    "config": OUTPUT_DIR / "phase2_model_data_config.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare train/test data and feature schema for phase-2 models."
    )
    parser.add_argument(
        "--input",
        default=str(INPUT_PATH),
        help=f"Input phase-2 model-ready table. Default: {INPUT_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help=f"Output directory. Default: {OUTPUT_DIR}",
    )
    parser.add_argument(
        "--allow-incomplete-target",
        action="store_true",
        help=(
            "Keep rows whose next-year target exists even when the next year is not "
            "marked complete. By default only complete next-year targets are kept."
        ),
    )
    return parser.parse_args()


def source_group_for_feature(feature: str) -> str:
    if feature == "year":
        return "time"
    if feature.startswith("annual_") or feature in {
        "parent_metro_region",
        "months_observed",
        "first_month",
        "last_month",
        "full_year_coverage",
        "months_with_median_sale_price",
        "months_with_median_list_price",
        "months_with_homes_sold",
        "months_with_pending_sales",
        "months_with_new_listings",
        "months_with_inventory",
    }:
        return "listings"
    if feature.startswith(("age_", "education_", "foreign_born", "native_born", "race_")):
        return "demographics"
    if feature in {
        "total_population",
        "log_total_population",
        "median_household_income",
        "log_median_household_income",
        "median_household_income_missing",
        "education_25plus_total",
        "foreign_born_total",
        "foreign_born_noncitizen_derived",
        "demographics_baseline_eligible",
        "male_share",
        "female_share",
        "born_other_state_25plus_share",
        "born_other_state_25plus_bachelors_or_higher_count",
        "born_other_state_25plus_bachelors_or_higher_share",
        "foreign_born_25plus_share",
        "foreign_born_25plus_bachelors_or_higher_count",
        "foreign_born_25plus_bachelors_or_higher_share",
    }:
        return "demographics"
    if feature.startswith(("avg_pct_", "median_pct_", "min_pct_", "max_pct_")) or feature in {
        "assessment_type",
        "is_caaspp",
        "is_cst",
        "school_year_complete",
        "school_quality_extended_eligible",
        "n_schools",
        "log1p_n_schools",
        "overall_score_range",
        "total_students_tested_reported_positive",
        "total_students_tested_unreliable",
    }:
        return "school_quality"
    if "minimum_wage" in feature or feature.startswith(("dominant_county", "zcta_county", "any_county")):
        return "minimum_wage"
    if feature.startswith(("crime_", "log1p_crime", "violent_component", "property_component")):
        return "crime"
    if feature.startswith(("amenity_", "amenities_")):
        return "amenities"
    if feature.startswith("phase2_has_"):
        return "feature_availability"
    return "other"


def feature_kind(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"


def load_phase2_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"zcta5": str}, low_memory=False)
    frame["zcta5"] = frame["zcta5"].astype(str).str.zfill(5)
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
    frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    frame[TARGET_ALIAS] = frame[TARGET_COLUMN]
    frame[LOG_TARGET_ALIAS] = np.where(
        frame[TARGET_ALIAS].gt(0),
        np.log(frame[TARGET_ALIAS]),
        np.nan,
    )
    return frame


def filter_valid_target(frame: pd.DataFrame, allow_incomplete_target: bool) -> pd.DataFrame:
    valid = frame[TARGET_ALIAS].notna() & frame[TARGET_ALIAS].gt(0)
    if not allow_incomplete_target:
        valid &= frame["target_next_year_complete"].eq(True)
    return frame[valid].copy()


def choose_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = set(IDENTIFIER_COLUMNS) | set(EXPLICIT_LEAKAGE_COLUMNS)
    excluded |= {TARGET_ALIAS, LOG_TARGET_ALIAS}
    return [col for col in frame.columns if col not in excluded]


def build_feature_schema(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in feature_columns:
        series = frame[col]
        rows.append(
            {
                "feature": col,
                "feature_kind": feature_kind(series),
                "source_group": source_group_for_feature(col),
                "missing_rate": float(series.isna().mean()),
                "n_unique": int(series.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows).sort_values(["source_group", "feature"]).reset_index(drop=True)


def split_train_test(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = frame[frame["year"].between(TRAIN_START_YEAR, TRAIN_END_YEAR)].copy()
    test = frame[frame["year"].eq(TEST_YEAR)].copy()
    train["split"] = "train_2012_2022"
    test["split"] = "test_2023"
    return train, test


def build_split_summary(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([train, test], ignore_index=True)
    summary = (
        combined.groupby(["split", "year"], as_index=False)
        .agg(
            rows=("zcta5", "size"),
            zctas=("zcta5", "nunique"),
            target_min=(TARGET_ALIAS, "min"),
            target_median=(TARGET_ALIAS, "median"),
            target_mean=(TARGET_ALIAS, "mean"),
            target_max=(TARGET_ALIAS, "max"),
        )
        .sort_values(["split", "year"])
    )
    return summary


def build_time_blocked_cv_folds(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold_id, validation_year in enumerate(range(2017, TRAIN_END_YEAR + 1), start=1):
        fold_train = train[train["year"].lt(validation_year)]
        fold_validation = train[train["year"].eq(validation_year)]
        rows.append(
            {
                "fold": fold_id,
                "validation_year": validation_year,
                "train_year_min": int(fold_train["year"].min()),
                "train_year_max": int(fold_train["year"].max()),
                "train_rows": int(len(fold_train)),
                "train_zctas": int(fold_train["zcta5"].nunique()),
                "validation_rows": int(len(fold_validation)),
                "validation_zctas": int(fold_validation["zcta5"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def write_outputs(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_schema: pd.DataFrame,
    split_summary: pd.DataFrame,
    cv_folds: pd.DataFrame,
    output_dir: Path,
    input_path: Path,
    allow_incomplete_target: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = {
        name: output_dir / path.name for name, path in OUTPUT_FILES.items()
    }

    train.to_csv(output_files["train"], index=False)
    test.to_csv(output_files["test"], index=False)
    feature_schema.to_csv(output_files["feature_schema"], index=False)
    split_summary.to_csv(output_files["split_summary"], index=False)
    cv_folds.to_csv(output_files["cv_folds"], index=False)

    config = {
        "input_path": str(input_path),
        "target_column": TARGET_COLUMN,
        "target_alias": TARGET_ALIAS,
        "log_target_alias": LOG_TARGET_ALIAS,
        "train_years": [TRAIN_START_YEAR, TRAIN_END_YEAR],
        "test_year": TEST_YEAR,
        "allow_incomplete_target": allow_incomplete_target,
        "require_complete_next_year_target": not allow_incomplete_target,
        "identifier_columns": sorted(IDENTIFIER_COLUMNS),
        "excluded_leakage_columns": sorted(EXPLICIT_LEAKAGE_COLUMNS),
        "n_features": int(len(feature_schema)),
        "feature_counts_by_kind": feature_schema["feature_kind"].value_counts().to_dict(),
        "feature_counts_by_source_group": feature_schema["source_group"].value_counts().to_dict(),
        "outputs": {key: str(value) for key, value in output_files.items()},
    }
    output_files["config"].write_text(json.dumps(config, indent=2))


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    phase2 = load_phase2_table(input_path)
    valid = filter_valid_target(phase2, allow_incomplete_target=args.allow_incomplete_target)
    train, test = split_train_test(valid)

    feature_columns = choose_feature_columns(valid)
    output_columns = [
        "zcta5",
        "year",
        "split",
        TARGET_ALIAS,
        LOG_TARGET_ALIAS,
        *[col for col in feature_columns if col not in {"year"}],
    ]
    train = train[output_columns].sort_values(["year", "zcta5"]).reset_index(drop=True)
    test = test[output_columns].sort_values(["year", "zcta5"]).reset_index(drop=True)

    model_rows = pd.concat([train, test], ignore_index=True)
    feature_schema = build_feature_schema(model_rows, feature_columns)
    split_summary = build_split_summary(train, test)
    cv_folds = build_time_blocked_cv_folds(train)

    write_outputs(
        train=train,
        test=test,
        feature_schema=feature_schema,
        split_summary=split_summary,
        cv_folds=cv_folds,
        output_dir=output_dir,
        input_path=input_path,
        allow_incomplete_target=args.allow_incomplete_target,
    )

    print(f"Prepared phase-2 model data from: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Train rows ({TRAIN_START_YEAR}-{TRAIN_END_YEAR}): {len(train):,}")
    print(f"Test rows ({TEST_YEAR} -> {TEST_YEAR + 1} target): {len(test):,}")
    print(f"Features retained: {len(feature_columns):,}")
    print("Feature kinds:")
    print(feature_schema["feature_kind"].value_counts().to_string())
    print("Feature groups:")
    print(feature_schema["source_group"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
