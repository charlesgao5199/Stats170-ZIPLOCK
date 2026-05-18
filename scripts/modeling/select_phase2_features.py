#!/usr/bin/env python3
"""Select the phase-2 modeling feature set and preprocessing groups."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DATA_DIR = PROJECT_ROOT / "outputs" / "phase2-models" / "model-data"
TRAIN_PATH = MODEL_DATA_DIR / "phase2_train_2012_2022.csv"
TEST_PATH = MODEL_DATA_DIR / "phase2_test_2023.csv"
FEATURE_SCHEMA_PATH = MODEL_DATA_DIR / "phase2_feature_schema.csv"

SELECTED_TRAIN_OUTPUT = MODEL_DATA_DIR / "phase2_selected_train_2012_2022.csv"
SELECTED_TEST_OUTPUT = MODEL_DATA_DIR / "phase2_selected_test_2023.csv"
SELECTED_SCHEMA_OUTPUT = MODEL_DATA_DIR / "phase2_selected_feature_schema.csv"
FEATURE_CONFIG_OUTPUT = MODEL_DATA_DIR / "phase2_selected_features.json"
SELECTION_SUMMARY_OUTPUT = MODEL_DATA_DIR / "phase2_feature_selection_summary.csv"
DUPLICATE_AUDIT_OUTPUT = MODEL_DATA_DIR / "phase2_duplicate_feature_audit.csv"

ID_AND_TARGET_COLUMNS = [
    "zcta5",
    "year",
    "split",
    "y_next_year_median_sale_price",
    "y_next_year_median_sale_price_log_change",
]

SELECTED_CRIME_FEATURES = {
    "crime_violent_est_per_1k",
    "crime_property_est_per_1k",
}

ARTIFACT_FEATURES = {
    "amenities_observed",
    "amenities_population_for_rate",
    "amenity_est_bucket_gap",
    "amenity_per_10k_bucket_gap",
    "crime_observed",
    "crime_counties",
    "crime_county_count",
    "crime_population_for_rate",
    "crime_allocation_ratio_sum",
    "crime_source_rows",
    "crime_source_total_double_counted_sum",
    "crime_multi_county_zcta",
    "crime_has_zero_or_missing_population",
    "violent_component_rounding_gap",
    "property_component_rounding_gap",
    "dominant_county_fips",
    "dominant_county_share",
    "zcta_county_area_share",
    "zcta_county_count",
    "minimum_wage_feature_method",
    "minimum_wage_dominant_county_geo_level",
    "minimum_wage_area_weighted_diff_annual_avg",
    "minimum_wage_county_area_weighted_annual_avg",
    "minimum_wage_county_area_weighted_year_start",
    "minimum_wage_dominant_county_annual_avg",
    "minimum_wage_dominant_county_year_start",
    "minimum_wage_for_model_year_start",
    "state_minimum_wage_annual_avg",
    "state_minimum_wage_year_start",
    "state_minimum_wage_source",
}

DUPLICATE_OR_CONSTANT_MINIMUM_WAGE_FEATURES = {
    "any_county_ordinance_above_state",
    "any_county_ordinance_record",
    "dominant_county_ordinance_above_state",
    "dominant_county_ordinance_record",
    "minimum_wage_local_premium_for_model_annual_avg",
    "phase2_has_minimum_wage",
}

RAW_LOG_DUPLICATES = {
    "median_household_income": "raw_duplicate_of_log_median_household_income",
    "total_population": "raw_duplicate_of_log_total_population",
    "n_schools": "raw_duplicate_of_log1p_n_schools",
}

DEMOGRAPHIC_COUNT_DUPLICATES = {
    "age_0_17_count",
    "age_18_24_count",
    "age_25_34_count",
    "age_35_44_count",
    "age_45_64_count",
    "age_65_plus_count",
    "born_other_state_25plus_bachelors_or_higher_count",
    "education_25plus_total",
    "education_bachelors_count",
    "education_bachelors_or_higher_count",
    "education_graduate_degree_count",
    "education_high_school_or_ged_count",
    "education_less_than_high_school_count",
    "education_some_college_or_associates_count",
    "foreign_born_25plus_bachelors_or_higher_count",
    "foreign_born_total",
}

COMPOSITION_REFERENCE_OR_COMPLEMENT_FEATURES = {
    "age_65_plus_share": "reference_category_for_age_share_composition",
    "education_bachelors_share": "nested_in_education_bachelors_or_higher_share",
    "education_graduate_degree_share": "nested_in_education_bachelors_or_higher_share",
    "education_high_school_or_ged_share": "reference_category_for_education_share_composition",
    "female_share": "exact_complement_of_male_share",
    "foreign_born_naturalized_share": "nested_in_foreign_born_total_share",
    "foreign_born_noncitizen_share": "nested_in_foreign_born_total_share",
    "foreign_born_naturalized_share_of_foreign_born": "exact_complement_of_foreign_born_noncitizen_share_of_foreign_born",
    "native_born_in_state_share": "reference_category_for_nativity_composition",
    "race_white_alone_share": "reference_category_for_race_share_composition",
}

LISTINGS_REDUNDANT_FEATURES = {
    "annual_median_list_price": "highly_collinear_with_annual_median_sale_price",
    "annual_median_list_ppsf": "highly_collinear_with_annual_median_ppsf",
    "annual_pending_sales": "highly_collinear_with_annual_homes_sold",
    "first_month": "coverage_artifact",
    "last_month": "coverage_artifact",
    "months_with_homes_sold": "exact_duplicate_of_months_observed_in_training_window",
    "months_with_inventory": "exact_duplicate_of_months_observed_in_training_window",
    "months_with_median_sale_price": "exact_duplicate_of_months_observed_in_training_window",
    "months_with_median_list_price": "near_duplicate_of_months_with_new_listings",
    "months_with_new_listings": "near_duplicate_of_months_with_median_list_price",
    "months_with_pending_sales": "coverage_artifact",
}

AMENITY_REDUNDANT_FEATURES = {
    "amenity_emp_1000_plus",
    "amenity_emp_100_249",
    "amenity_emp_10_19",
    "amenity_emp_1_4",
    "amenity_emp_20_49",
    "amenity_emp_250_499",
    "amenity_emp_500_999",
    "amenity_emp_50_99",
    "amenity_emp_5_9",
    "amenity_employee_bucket_total",
    "amenity_est_coffee_shops",
    "amenity_est_fitness",
    "amenity_est_parks",
    "amenity_est_restaurants",
    "amenity_est_total",
    "amenity_per_10k_total",
    "amenity_source_rows",
}

SCHOOL_REDUNDANT_FEATURES = {
    "avg_pct_met_ela",
    "avg_pct_met_ela_pct_assessment_year",
    "avg_pct_met_math",
    "avg_pct_met_math_pct_assessment_year",
    "avg_pct_met_overall",
    "avg_pct_met_overall_pct_assessment_year",
    "is_caaspp",
    "is_cst",
    "max_pct_met_overall",
    "median_pct_met_overall",
    "median_pct_met_overall_pct_assessment_year",
    "min_pct_met_overall",
    "school_quality_extended_eligible",
    "total_students_tested_reported_positive",
}

TIME_OR_GEO_DUPLICATE_FEATURES = {
    "year": "time_trend_feature_kept_only_as_split_metadata",
    "dominant_county": "geographic_duplicate_of_parent_metro_region",
}

FEATURE_AVAILABILITY_FLAGS = {
    "phase2_has_amenities",
    "phase2_has_crime",
    "phase2_has_demographics",
    "phase2_has_school_quality",
}

HIGH_CORRELATION_DUPLICATES = {
    "born_other_state_25plus_share": "highly_collinear_with_native_born_other_state_us_share",
    "foreign_born_25plus_share": "highly_collinear_with_foreign_born_total_share",
    "annual_inventory_end": "highly_collinear_with_annual_inventory_mean",
    "annual_new_listings": "highly_collinear_with_annual_homes_sold",
    "avg_pct_met_ela_z_assessment_year": "highly_collinear_with_avg_pct_met_overall_z_assessment_year",
    "avg_pct_met_math_z_assessment_year": "highly_collinear_with_avg_pct_met_overall_z_assessment_year",
    "median_pct_met_overall_z_assessment_year": "highly_collinear_with_avg_pct_met_overall_z_assessment_year",
}


def load_model_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_PATH, dtype={"zcta5": str}, low_memory=False)
    test = pd.read_csv(TEST_PATH, dtype={"zcta5": str}, low_memory=False)
    schema = pd.read_csv(FEATURE_SCHEMA_PATH)
    return train, test, schema


def normalize_bool_like(series: pd.Series) -> bool:
    nonnull = series.dropna()
    if nonnull.empty:
        return False
    if pd.api.types.is_bool_dtype(nonnull):
        return True
    values = {str(value).strip().lower() for value in nonnull.unique()}
    return values.issubset({"true", "false", "0", "1"})


def preprocessing_kind(feature: str, train: pd.DataFrame, test: pd.DataFrame) -> str:
    combined = pd.concat([train[[feature]], test[[feature]]], ignore_index=True)[feature]
    if normalize_bool_like(combined):
        return "boolean"
    if pd.api.types.is_numeric_dtype(combined):
        return "numeric"
    return "categorical"


def exclusion_reason(row: pd.Series) -> str | None:
    feature = row["feature"]
    source_group = row["source_group"]

    if feature in RAW_LOG_DUPLICATES:
        return RAW_LOG_DUPLICATES[feature]
    if feature in DEMOGRAPHIC_COUNT_DUPLICATES:
        return "count_duplicate_of_share_or_log_population_feature"
    if feature in COMPOSITION_REFERENCE_OR_COMPLEMENT_FEATURES:
        return COMPOSITION_REFERENCE_OR_COMPLEMENT_FEATURES[feature]
    if feature in LISTINGS_REDUNDANT_FEATURES:
        return LISTINGS_REDUNDANT_FEATURES[feature]
    if feature in AMENITY_REDUNDANT_FEATURES:
        return "amenity_raw_count_or_employee_bucket_duplicate_of_per_capita_density"
    if feature in SCHOOL_REDUNDANT_FEATURES:
        return "school_raw_or_percentile_duplicate_of_selected_z_score_or_quality_flag"
    if feature in TIME_OR_GEO_DUPLICATE_FEATURES:
        return TIME_OR_GEO_DUPLICATE_FEATURES[feature]
    if feature in FEATURE_AVAILABILITY_FLAGS:
        return "feature_availability_flag_not_substantive_predictor"
    if feature in HIGH_CORRELATION_DUPLICATES:
        return HIGH_CORRELATION_DUPLICATES[feature]
    if feature in SELECTED_CRIME_FEATURES:
        return None
    if source_group == "crime":
        return "crime_restricted_to_violent_and_property_rates"
    if feature in ARTIFACT_FEATURES:
        return "processing_artifact_or_duplicate_source_field"
    if feature in DUPLICATE_OR_CONSTANT_MINIMUM_WAGE_FEATURES:
        return "constant_or_duplicate_minimum_wage_signal_in_train_test_window"
    if int(row["n_unique"]) <= 1 and float(row["missing_rate"]) == 0:
        return "constant_in_train_test_window"
    return None


def build_selected_schema(
    train: pd.DataFrame,
    test: pd.DataFrame,
    schema: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, row in schema.iterrows():
        reason = exclusion_reason(row)
        feature = row["feature"]
        rows.append(
            {
                **row.to_dict(),
                "selected": reason is None,
                "exclusion_reason": reason or "",
                "preprocessing_kind": preprocessing_kind(feature, train, test),
            }
        )
    selected_schema = pd.DataFrame(rows)
    return selected_schema.sort_values(
        ["selected", "source_group", "feature"], ascending=[False, True, True]
    ).reset_index(drop=True)


def build_feature_sets(selected_schema: pd.DataFrame) -> dict[str, list[str]]:
    selected = selected_schema[selected_schema["selected"]]

    def by_groups(groups: set[str]) -> list[str]:
        return selected[selected["source_group"].isin(groups)]["feature"].tolist()

    baseline_groups = {"listings", "demographics", "time", "feature_availability"}
    phase2_groups = {
        "listings",
        "demographics",
        "time",
        "feature_availability",
        "school_quality",
        "minimum_wage",
        "crime",
        "amenities",
        "other",
    }
    neighborhood_groups = {"school_quality", "crime", "amenities"}
    policy_groups = {"minimum_wage"}

    return {
        "all_selected_features": selected["feature"].tolist(),
        "baseline_features": by_groups(baseline_groups),
        "phase2_features": by_groups(phase2_groups),
        "neighborhood_features": by_groups(neighborhood_groups),
        "policy_features": by_groups(policy_groups),
        "crime_features": selected[selected["source_group"].eq("crime")]["feature"].tolist(),
    }


def build_config(selected_schema: pd.DataFrame) -> dict[str, object]:
    selected = selected_schema[selected_schema["selected"]].copy()
    excluded = selected_schema[~selected_schema["selected"]].copy()
    feature_sets = build_feature_sets(selected_schema)

    preprocessing = {
        kind: selected[selected["preprocessing_kind"].eq(kind)]["feature"].tolist()
        for kind in ["numeric", "categorical", "boolean"]
    }
    source_groups = {
        group: group_df["feature"].tolist()
        for group, group_df in selected.groupby("source_group", sort=True)
    }
    exclusions = {
        reason: group_df["feature"].tolist()
        for reason, group_df in excluded.groupby("exclusion_reason", sort=True)
    }

    return {
        "target_columns": {
            "price": "y_next_year_median_sale_price",
            "log_change": "y_next_year_median_sale_price_log_change",
        },
        "selected_feature_count": int(len(selected)),
        "excluded_feature_count": int(len(excluded)),
        "preprocessing": preprocessing,
        "source_groups": source_groups,
        "feature_sets": feature_sets,
        "exclusions": exclusions,
        "notes": [
            "Crime is restricted to violent and property crime rates per 1,000 residents.",
            "Nested crime components are excluded because they aggregate into violent/property totals.",
            "Minimum wage keeps the annual average model wage and removes duplicate state/county variants.",
            "Feature availability flags are excluded because they proxy data coverage rather than substantive predictors.",
        ],
    }


def write_selected_data(
    train: pd.DataFrame,
    test: pd.DataFrame,
    selected_features: list[str],
) -> None:
    columns = [*ID_AND_TARGET_COLUMNS, *[feature for feature in selected_features if feature not in ID_AND_TARGET_COLUMNS]]
    train[columns].to_csv(SELECTED_TRAIN_OUTPUT, index=False)
    test[columns].to_csv(SELECTED_TEST_OUTPUT, index=False)


def write_summary(selected_schema: pd.DataFrame) -> None:
    summary = (
        selected_schema.groupby(["source_group", "selected"], as_index=False)
        .agg(features=("feature", "count"))
        .sort_values(["source_group", "selected"])
    )
    summary.to_csv(SELECTION_SUMMARY_OUTPUT, index=False)


def write_duplicate_audit(train: pd.DataFrame, selected_schema: pd.DataFrame) -> None:
    selected_features = selected_schema[selected_schema["selected"]]["feature"].tolist()
    selected_numeric = [
        feature
        for feature in selected_features
        if feature in train.columns and pd.api.types.is_numeric_dtype(train[feature])
    ]
    rows = []
    if len(selected_numeric) >= 2:
        corr = train[selected_numeric].corr(numeric_only=True).abs()
        for i, left in enumerate(selected_numeric):
            for right in selected_numeric[i + 1 :]:
                value = corr.loc[left, right]
                if pd.notna(value) and value >= 0.95:
                    rows.append(
                        {
                            "left_feature": left,
                            "right_feature": right,
                            "abs_correlation": float(value),
                            "audit_note": "selected_numeric_pair_abs_corr_ge_0.95",
                        }
                    )
    audit = pd.DataFrame(
        rows,
        columns=["left_feature", "right_feature", "abs_correlation", "audit_note"],
    )
    if not audit.empty:
        audit = audit.sort_values("abs_correlation", ascending=False)
    audit.to_csv(DUPLICATE_AUDIT_OUTPUT, index=False)


def main() -> int:
    train, test, schema = load_model_data()
    selected_schema = build_selected_schema(train, test, schema)
    config = build_config(selected_schema)
    selected_features = config["feature_sets"]["all_selected_features"]

    write_selected_data(train, test, selected_features)
    selected_schema.to_csv(SELECTED_SCHEMA_OUTPUT, index=False)
    FEATURE_CONFIG_OUTPUT.write_text(json.dumps(config, indent=2))
    write_summary(selected_schema)
    write_duplicate_audit(train, selected_schema)

    print(f"Wrote selected training data: {SELECTED_TRAIN_OUTPUT}")
    print(f"Wrote selected test data: {SELECTED_TEST_OUTPUT}")
    print(f"Wrote selected feature schema: {SELECTED_SCHEMA_OUTPUT}")
    print(f"Wrote selected feature config: {FEATURE_CONFIG_OUTPUT}")
    print(f"Wrote duplicate audit: {DUPLICATE_AUDIT_OUTPUT}")
    print(f"Selected features: {config['selected_feature_count']:,}")
    print(f"Excluded features: {config['excluded_feature_count']:,}")
    print("Selected by preprocessing kind:")
    for kind, features in config["preprocessing"].items():
        print(f"  {kind}: {len(features):,}")
    print("Selected crime features:")
    for feature in config["feature_sets"]["crime_features"]:
        print(f"  {feature}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
