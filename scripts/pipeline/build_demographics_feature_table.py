#!/usr/bin/env python3
"""Build cleaned and feature-oriented annual ZCTA demographics tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "model-datasets" / "zcta_demographics"
OUTPUT_DIR = PROJECT_ROOT / "model-ready"
CLEAN_OUTPUT = OUTPUT_DIR / "zcta_demographics_cleaned.csv"
FEATURE_OUTPUT = OUTPUT_DIR / "zcta_demographics_features.csv"

ID_COLUMNS = ["year", "zcta5", "geo_id", "zcta_name"]
SPECIAL_MISSING_VALUES = {-666666666, -888888888, -999999999}

AGE_BUCKETS = {
    "age_0_17": [
        "male_age_under_5",
        "male_age_5_to_9",
        "male_age_10_to_14",
        "male_age_15_to_17",
        "female_age_under_5",
        "female_age_5_to_9",
        "female_age_10_to_14",
        "female_age_15_to_17",
    ],
    "age_18_24": [
        "male_age_18_to_19",
        "male_age_20",
        "male_age_21",
        "male_age_22_to_24",
        "female_age_18_to_19",
        "female_age_20",
        "female_age_21",
        "female_age_22_to_24",
    ],
    "age_25_34": [
        "male_age_25_to_29",
        "male_age_30_to_34",
        "female_age_25_to_29",
        "female_age_30_to_34",
    ],
    "age_35_44": [
        "male_age_35_to_39",
        "male_age_40_to_44",
        "female_age_35_to_39",
        "female_age_40_to_44",
    ],
    "age_45_64": [
        "male_age_45_to_49",
        "male_age_50_to_54",
        "male_age_55_to_59",
        "male_age_60_to_61",
        "male_age_62_to_64",
        "female_age_45_to_49",
        "female_age_50_to_54",
        "female_age_55_to_59",
        "female_age_60_to_61",
        "female_age_62_to_64",
    ],
    "age_65_plus": [
        "male_age_65_to_66",
        "male_age_67_to_69",
        "male_age_70_to_74",
        "male_age_75_to_79",
        "male_age_80_to_84",
        "male_age_85_plus",
        "female_age_65_to_66",
        "female_age_67_to_69",
        "female_age_70_to_74",
        "female_age_75_to_79",
        "female_age_80_to_84",
        "female_age_85_plus",
    ],
}


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Return NaN when the denominator is null or non-positive."""
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    result = pd.Series(np.nan, index=numerator.index, dtype="float64")
    valid = denominator.notna() & numerator.notna() & denominator.gt(0)
    result.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    return result


def sum_columns(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    return df[columns].sum(axis=1, min_count=1)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH, dtype=str, low_memory=False)

    numeric_columns = [c for c in df.columns if c not in ID_COLUMNS]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col].isin(SPECIAL_MISSING_VALUES), col] = np.nan

    # Fill structurally missing early-year noncitizen counts from total - naturalized.
    noncit_missing = df["foreign_born_noncitizen"].isna()
    df["foreign_born_noncitizen_derived"] = False
    df.loc[noncit_missing, "foreign_born_noncitizen"] = (
        df.loc[noncit_missing, "foreign_born_total"] - df.loc[noncit_missing, "foreign_born_naturalized"]
    )
    df.loc[noncit_missing & df["foreign_born_noncitizen"].notna(), "foreign_born_noncitizen_derived"] = True

    # Guard against any tiny negative values from bad source math.
    df.loc[df["foreign_born_noncitizen"] < 0, "foreign_born_noncitizen"] = np.nan

    df["population_nonzero"] = df["total_population"].gt(0)
    df["median_household_income_missing"] = df["median_household_income"].isna()
    df["demographics_baseline_eligible"] = (
        df["population_nonzero"] & df["education_25plus_total"].gt(0)
    )

    # Persist a cleaned raw annual table.
    clean = df.copy()
    clean["year"] = clean["year"].astype(int)
    clean["zcta5"] = clean["zcta5"].astype(str).str.zfill(5)
    clean = clean.sort_values(["zcta5", "year"]).reset_index(drop=True)
    clean.to_csv(CLEAN_OUTPUT, index=False)

    # Feature-oriented table for modeling / merge.
    features = clean[clean["population_nonzero"]].copy()

    features["log_total_population"] = np.log(features["total_population"])
    features["log_median_household_income"] = np.where(
        features["median_household_income"].gt(0),
        np.log(features["median_household_income"]),
        np.nan,
    )

    features["male_share"] = safe_divide(features["male_total"], features["total_population"])
    features["female_share"] = safe_divide(features["female_total"], features["total_population"])

    for bucket_name, columns in AGE_BUCKETS.items():
        features[f"{bucket_name}_count"] = sum_columns(features, columns)
        features[f"{bucket_name}_share"] = safe_divide(
            features[f"{bucket_name}_count"], features["total_population"]
        )

    features["education_less_than_high_school_count"] = sum_columns(
        features,
        [
            "education_25plus_no_schooling",
            "education_25plus_nursery_school",
            "education_25plus_kindergarten",
            "education_25plus_grade_1",
            "education_25plus_grade_2",
            "education_25plus_grade_3",
            "education_25plus_grade_4",
            "education_25plus_grade_5",
            "education_25plus_grade_6",
            "education_25plus_grade_7",
            "education_25plus_grade_8",
            "education_25plus_grade_9",
            "education_25plus_grade_10",
            "education_25plus_grade_11",
            "education_25plus_grade_12_no_diploma",
        ],
    )
    features["education_high_school_or_ged_count"] = sum_columns(
        features,
        ["education_25plus_high_school_diploma", "education_25plus_ged"],
    )
    features["education_some_college_or_associates_count"] = sum_columns(
        features,
        [
            "education_25plus_some_college_less_than_1_year",
            "education_25plus_some_college_1plus_years_no_degree",
            "education_25plus_associates",
        ],
    )
    features["education_bachelors_count"] = features["education_25plus_bachelors"]
    features["education_graduate_degree_count"] = sum_columns(
        features,
        [
            "education_25plus_masters",
            "education_25plus_professional_school",
            "education_25plus_doctorate",
        ],
    )
    features["education_bachelors_or_higher_count"] = (
        features["education_bachelors_count"] + features["education_graduate_degree_count"]
    )

    education_share_bases = [
        "education_less_than_high_school",
        "education_high_school_or_ged",
        "education_some_college_or_associates",
        "education_bachelors",
        "education_graduate_degree",
        "education_bachelors_or_higher",
    ]
    for base in education_share_bases:
        features[f"{base}_share"] = safe_divide(
            features[f"{base}_count"], features["education_25plus_total"]
        )

    nativity_bases = [
        "native_born_in_state",
        "native_born_other_state_us",
        "foreign_born_total",
        "foreign_born_naturalized",
        "foreign_born_noncitizen",
    ]
    for base in nativity_bases:
        features[f"{base}_share"] = safe_divide(features[base], features["total_population"])

    features["foreign_born_naturalized_share_of_foreign_born"] = safe_divide(
        features["foreign_born_naturalized"], features["foreign_born_total"]
    )
    features["foreign_born_noncitizen_share_of_foreign_born"] = safe_divide(
        features["foreign_born_noncitizen"], features["foreign_born_total"]
    )

    features["born_other_state_25plus_share"] = safe_divide(
        features["born_other_state_25plus_total"], features["education_25plus_total"]
    )
    features["born_other_state_25plus_bachelors_or_higher_count"] = (
        features["born_other_state_25plus_bachelors"] + features["born_other_state_25plus_graduate_degree"]
    )
    features["born_other_state_25plus_bachelors_or_higher_share"] = safe_divide(
        features["born_other_state_25plus_bachelors_or_higher_count"],
        features["born_other_state_25plus_total"],
    )
    features["foreign_born_25plus_share"] = safe_divide(
        features["foreign_born_25plus_total"], features["education_25plus_total"]
    )
    features["foreign_born_25plus_bachelors_or_higher_count"] = (
        features["foreign_born_25plus_bachelors"] + features["foreign_born_25plus_graduate_degree"]
    )
    features["foreign_born_25plus_bachelors_or_higher_share"] = safe_divide(
        features["foreign_born_25plus_bachelors_or_higher_count"],
        features["foreign_born_25plus_total"],
    )

    race_columns = [
        "race_white_alone",
        "race_black_alone",
        "race_american_indian_or_alaska_native_alone",
        "race_asian_alone",
        "race_native_hawaiian_or_pacific_islander_alone",
        "race_some_other_race_alone",
        "race_two_or_more",
        "hispanic_or_latino_total",
    ]
    for col in race_columns:
        features[f"{col}_share"] = safe_divide(features[col], features["total_population"])

    feature_columns = [
        "year",
        "zcta5",
        "geo_id",
        "zcta_name",
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
    ]
    feature_columns += [f"{bucket}_count" for bucket in AGE_BUCKETS]
    feature_columns += [f"{bucket}_share" for bucket in AGE_BUCKETS]
    feature_columns += [
        "education_less_than_high_school_count",
        "education_less_than_high_school_share",
        "education_high_school_or_ged_count",
        "education_high_school_or_ged_share",
        "education_some_college_or_associates_count",
        "education_some_college_or_associates_share",
        "education_bachelors_count",
        "education_bachelors_share",
        "education_graduate_degree_count",
        "education_graduate_degree_share",
        "education_bachelors_or_higher_count",
        "education_bachelors_or_higher_share",
        "native_born_in_state_share",
        "native_born_other_state_us_share",
        "foreign_born_total_share",
        "foreign_born_naturalized_share",
        "foreign_born_noncitizen_share",
        "foreign_born_naturalized_share_of_foreign_born",
        "foreign_born_noncitizen_share_of_foreign_born",
        "born_other_state_25plus_share",
        "born_other_state_25plus_bachelors_or_higher_count",
        "born_other_state_25plus_bachelors_or_higher_share",
        "foreign_born_25plus_share",
        "foreign_born_25plus_bachelors_or_higher_count",
        "foreign_born_25plus_bachelors_or_higher_share",
        "race_white_alone_share",
        "race_black_alone_share",
        "race_american_indian_or_alaska_native_alone_share",
        "race_asian_alone_share",
        "race_native_hawaiian_or_pacific_islander_alone_share",
        "race_some_other_race_alone_share",
        "race_two_or_more_share",
        "hispanic_or_latino_total_share",
    ]

    features = features[feature_columns].sort_values(["zcta5", "year"]).reset_index(drop=True)
    features.to_csv(FEATURE_OUTPUT, index=False)

    print(f"Wrote cleaned demographics table: {CLEAN_OUTPUT}")
    print(f"Rows: {len(clean):,}")
    print(f"Columns: {len(clean.columns):,}")
    print()
    print(f"Wrote demographics feature table: {FEATURE_OUTPUT}")
    print(f"Rows: {len(features):,}")
    print(f"Columns: {len(features.columns):,}")
    print(f"Rows with nonmissing income: {int(features['median_household_income_missing'].eq(False).sum()):,}")
    print(f"Rows with derived noncitizen counts: {int(features['foreign_born_noncitizen_derived'].sum()):,}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
