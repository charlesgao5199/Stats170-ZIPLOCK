#!/usr/bin/env python3
"""Build ZCTA-year minimum wage features from city/county wage records."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "model-datasets" / "minimum_wage_updated"
ZCTA_COUNTY_RELATIONSHIP_PATH = (
    PROJECT_ROOT / "data" / "zcta_archive" / "source" / "2020_ZCTA_relationship.txt"
)
MODEL_READY_DIR = PROJECT_ROOT / "model-ready"

CLEAN_OUTPUT = MODEL_READY_DIR / "minimum_wage_city_county_cleaned.csv"
CROSSWALK_OUTPUT = MODEL_READY_DIR / "zcta_county_crosswalk.csv"
FEATURE_OUTPUT = MODEL_READY_DIR / "zcta_minimum_wage_features.csv"
MERGED_LISTINGS_OUTPUT = MODEL_READY_DIR / "listings_model_ready_with_minimum_wage.csv"

CALIFORNIA_FIPS_PREFIX = "06"

# California DIR history: $8.00 from Jan. 1, 2008; $9.00 from Jul. 1, 2014;
# $10.00 from Jan. 1, 2016. The input table supplies state-law rates for 2016+.
PRE_INPUT_STATE_YEAR_START_WAGE = {
    2012: 8.00,
    2013: 8.00,
    2014: 8.00,
    2015: 9.00,
}
PRE_INPUT_STATE_ANNUAL_AVG_WAGE = {
    2012: 8.00,
    2013: 8.00,
    2014: ((8.00 * 181) + (9.00 * 184)) / 365,
    2015: 9.00,
}


def normalize_county_name(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\s+County$", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
    )


def load_minimum_wage() -> pd.DataFrame:
    wage = pd.read_csv(INPUT_PATH)
    wage = wage.rename(
        columns={
            "City": "city",
            "County": "county",
            "Year": "year",
            "Minimum Wage": "minimum_wage_year_start",
            "Ordinance Type": "ordinance_type",
        }
    )
    wage["city"] = wage["city"].astype(str).str.strip()
    wage["county"] = normalize_county_name(wage["county"])
    wage["year"] = pd.to_numeric(wage["year"], errors="raise").astype(int)
    wage["minimum_wage_year_start"] = pd.to_numeric(
        wage["minimum_wage_year_start"], errors="raise"
    )
    wage["minimum_wage_annual_avg"] = wage["minimum_wage_year_start"]
    wage["minimum_wage_backfilled"] = False
    wage["minimum_wage_source_detail"] = "input_minimum_wage_updated"
    return wage


def build_state_schedule(wage: pd.DataFrame) -> pd.DataFrame:
    state_from_input = (
        wage[wage["ordinance_type"].eq("State Law")]
        .groupby("year", as_index=False)["minimum_wage_year_start"]
        .first()
    )
    state_from_input["minimum_wage_annual_avg"] = state_from_input[
        "minimum_wage_year_start"
    ]

    pre_input = pd.DataFrame(
        {
            "year": sorted(PRE_INPUT_STATE_YEAR_START_WAGE),
            "minimum_wage_year_start": [
                PRE_INPUT_STATE_YEAR_START_WAGE[year]
                for year in sorted(PRE_INPUT_STATE_YEAR_START_WAGE)
            ],
            "minimum_wage_annual_avg": [
                PRE_INPUT_STATE_ANNUAL_AVG_WAGE[year]
                for year in sorted(PRE_INPUT_STATE_ANNUAL_AVG_WAGE)
            ],
        }
    )

    state = pd.concat([pre_input, state_from_input], ignore_index=True)
    state = state.drop_duplicates(subset=["year"], keep="first").sort_values("year")
    state = state.rename(
        columns={
            "minimum_wage_year_start": "state_minimum_wage_year_start",
            "minimum_wage_annual_avg": "state_minimum_wage_annual_avg",
        }
    )
    state["state_minimum_wage_source"] = np.where(
        state["year"].lt(wage["year"].min()),
        "california_dir_history_backfill",
        "input_state_law_rows",
    )
    return state


def backfill_city_county_wage(wage: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
    locations = wage[["city", "county"]].drop_duplicates().sort_values(["county", "city"])
    pre_years = state[state["year"].lt(wage["year"].min())].copy()
    pre_years = pre_years.rename(
        columns={
            "state_minimum_wage_year_start": "minimum_wage_year_start",
            "state_minimum_wage_annual_avg": "minimum_wage_annual_avg",
        }
    )

    backfilled = locations.merge(pre_years, how="cross")
    backfilled["ordinance_type"] = "State Law"
    backfilled["minimum_wage_backfilled"] = True
    backfilled["minimum_wage_source_detail"] = "state_law_backfill_pre_input_years"

    clean = pd.concat(
        [
            backfilled[
                [
                    "city",
                    "county",
                    "year",
                    "minimum_wage_year_start",
                    "minimum_wage_annual_avg",
                    "ordinance_type",
                    "minimum_wage_backfilled",
                    "minimum_wage_source_detail",
                ]
            ],
            wage[
                [
                    "city",
                    "county",
                    "year",
                    "minimum_wage_year_start",
                    "minimum_wage_annual_avg",
                    "ordinance_type",
                    "minimum_wage_backfilled",
                    "minimum_wage_source_detail",
                ]
            ],
        ],
        ignore_index=True,
    )
    clean = clean.sort_values(["county", "city", "year"]).reset_index(drop=True)
    return clean


def build_county_year_wage(clean: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
    county_ordinance = (
        clean[clean["ordinance_type"].eq("County Ordinance")]
        .groupby(["county", "year"], as_index=False)
        .agg(
            county_ordinance_wage_year_start=("minimum_wage_year_start", "max"),
            county_ordinance_wage_annual_avg=("minimum_wage_annual_avg", "max"),
            county_ordinance_city_rows=("city", "nunique"),
        )
    )

    counties = clean[["county"]].drop_duplicates().sort_values("county")
    years = state[["year"]].drop_duplicates().sort_values("year")
    county_year = counties.merge(years, how="cross").merge(
        county_ordinance, on=["county", "year"], how="left"
    )
    county_year = county_year.merge(state, on="year", how="left", validate="many_to_one")

    county_year["effective_minimum_wage_year_start"] = county_year[
        ["state_minimum_wage_year_start", "county_ordinance_wage_year_start"]
    ].max(axis=1)
    county_year["effective_minimum_wage_annual_avg"] = county_year[
        ["state_minimum_wage_annual_avg", "county_ordinance_wage_annual_avg"]
    ].max(axis=1)
    county_year["county_ordinance_record"] = county_year[
        "county_ordinance_wage_year_start"
    ].notna()
    county_year["county_ordinance_above_state"] = county_year[
        "county_ordinance_wage_annual_avg"
    ].gt(county_year["state_minimum_wage_annual_avg"])
    county_year["minimum_wage_geo_level"] = np.where(
        county_year["county_ordinance_above_state"], "County Ordinance", "State Law"
    )
    county_year["local_minimum_wage_premium_annual_avg"] = (
        county_year["effective_minimum_wage_annual_avg"]
        - county_year["state_minimum_wage_annual_avg"]
    )
    county_year["minimum_wage_backfilled"] = county_year["year"].lt(
        clean.loc[~clean["minimum_wage_backfilled"], "year"].min()
    )
    return county_year.sort_values(["county", "year"]).reset_index(drop=True)


def build_zcta_county_crosswalk() -> pd.DataFrame:
    relationship = pd.read_csv(
        ZCTA_COUNTY_RELATIONSHIP_PATH,
        sep="|",
        dtype=str,
        low_memory=False,
    )
    relationship = relationship[
        relationship["GEOID_ZCTA5_20"].notna()
        & relationship["GEOID_COUNTY_20"].str.startswith(CALIFORNIA_FIPS_PREFIX, na=False)
    ].copy()

    relationship["zcta5"] = relationship["GEOID_ZCTA5_20"].astype(str).str.zfill(5)
    relationship["county_fips"] = relationship["GEOID_COUNTY_20"].astype(str).str.zfill(5)
    relationship["county"] = normalize_county_name(relationship["NAMELSAD_COUNTY_20"])
    relationship["area_land_part"] = pd.to_numeric(
        relationship["AREALAND_PART"], errors="coerce"
    ).fillna(0)
    relationship["area_water_part"] = pd.to_numeric(
        relationship["AREAWATER_PART"], errors="coerce"
    ).fillna(0)
    relationship["area_part"] = relationship["area_land_part"]
    zero_land = relationship.groupby("zcta5")["area_part"].transform("sum").eq(0)
    relationship.loc[zero_land, "area_part"] = (
        relationship.loc[zero_land, "area_land_part"]
        + relationship.loc[zero_land, "area_water_part"]
    )

    relationship["zcta_total_area_part"] = relationship.groupby("zcta5")[
        "area_part"
    ].transform("sum")
    relationship["zcta_county_area_share"] = np.where(
        relationship["zcta_total_area_part"].gt(0),
        relationship["area_part"] / relationship["zcta_total_area_part"],
        np.nan,
    )
    relationship["zcta_county_count"] = relationship.groupby("zcta5")[
        "county_fips"
    ].transform("nunique")
    relationship = relationship.sort_values(
        ["zcta5", "zcta_county_area_share", "county_fips"],
        ascending=[True, False, True],
    )
    relationship["county_rank_in_zcta"] = relationship.groupby("zcta5").cumcount() + 1
    relationship["is_dominant_county"] = relationship["county_rank_in_zcta"].eq(1)
    relationship["dominant_county_share"] = relationship.groupby("zcta5")[
        "zcta_county_area_share"
    ].transform("max")
    relationship["zcta_county_assignment_quality"] = np.select(
        [
            relationship["zcta_county_count"].eq(1),
            relationship["dominant_county_share"].ge(0.8),
        ],
        ["single_county", "dominant_county_ge_80pct"],
        default="multi_county_low_dominance",
    )

    return relationship[
        [
            "zcta5",
            "county_fips",
            "county",
            "zcta_county_area_share",
            "zcta_county_count",
            "county_rank_in_zcta",
            "is_dominant_county",
            "dominant_county_share",
            "zcta_county_assignment_quality",
        ]
    ].reset_index(drop=True)


def build_zcta_features(
    crosswalk: pd.DataFrame,
    county_year: pd.DataFrame,
    state: pd.DataFrame,
) -> pd.DataFrame:
    years = state[["year"]].drop_duplicates().sort_values("year")
    dominant = crosswalk[crosswalk["is_dominant_county"]].copy()

    dominant_features = dominant.merge(years, how="cross").merge(
        county_year,
        on=["county", "year"],
        how="left",
        validate="many_to_one",
    )
    dominant_features = dominant_features.rename(
        columns={
            "county": "dominant_county",
            "county_fips": "dominant_county_fips",
            "effective_minimum_wage_year_start": "minimum_wage_dominant_county_year_start",
            "effective_minimum_wage_annual_avg": "minimum_wage_dominant_county_annual_avg",
            "minimum_wage_geo_level": "minimum_wage_dominant_county_geo_level",
            "county_ordinance_record": "dominant_county_ordinance_record",
            "county_ordinance_above_state": "dominant_county_ordinance_above_state",
            "local_minimum_wage_premium_annual_avg": "dominant_county_local_premium_annual_avg",
        }
    )

    weighted = crosswalk.merge(years, how="cross").merge(
        county_year,
        on=["county", "year"],
        how="left",
        validate="many_to_one",
    )
    weighted["weighted_wage_year_start"] = (
        weighted["effective_minimum_wage_year_start"] * weighted["zcta_county_area_share"]
    )
    weighted["weighted_wage_annual_avg"] = (
        weighted["effective_minimum_wage_annual_avg"] * weighted["zcta_county_area_share"]
    )
    weighted_features = (
        weighted.groupby(["zcta5", "year"], as_index=False)
        .agg(
            minimum_wage_county_area_weighted_year_start=(
                "weighted_wage_year_start",
                "sum",
            ),
            minimum_wage_county_area_weighted_annual_avg=(
                "weighted_wage_annual_avg",
                "sum",
            ),
            any_county_ordinance_record=("county_ordinance_record", "max"),
            any_county_ordinance_above_state=("county_ordinance_above_state", "max"),
        )
    )

    feature = dominant_features.merge(
        weighted_features, on=["zcta5", "year"], how="left", validate="one_to_one"
    )
    feature["minimum_wage_feature_method"] = "dominant_county_with_area_weighted_check"
    feature["minimum_wage_for_model_annual_avg"] = feature[
        "minimum_wage_dominant_county_annual_avg"
    ]
    feature["minimum_wage_for_model_year_start"] = feature[
        "minimum_wage_dominant_county_year_start"
    ]
    feature["minimum_wage_local_premium_for_model_annual_avg"] = (
        feature["minimum_wage_for_model_annual_avg"]
        - feature["state_minimum_wage_annual_avg"]
    )
    feature["minimum_wage_area_weighted_diff_annual_avg"] = (
        feature["minimum_wage_county_area_weighted_annual_avg"]
        - feature["minimum_wage_for_model_annual_avg"]
    )
    feature["minimum_wage_imputed"] = feature["minimum_wage_backfilled"]

    keep_columns = [
        "zcta5",
        "year",
        "state_minimum_wage_year_start",
        "state_minimum_wage_annual_avg",
        "state_minimum_wage_source",
        "minimum_wage_for_model_year_start",
        "minimum_wage_for_model_annual_avg",
        "minimum_wage_local_premium_for_model_annual_avg",
        "minimum_wage_dominant_county_year_start",
        "minimum_wage_dominant_county_annual_avg",
        "minimum_wage_county_area_weighted_year_start",
        "minimum_wage_county_area_weighted_annual_avg",
        "minimum_wage_area_weighted_diff_annual_avg",
        "minimum_wage_dominant_county_geo_level",
        "minimum_wage_feature_method",
        "minimum_wage_imputed",
        "dominant_county",
        "dominant_county_fips",
        "zcta_county_area_share",
        "zcta_county_count",
        "dominant_county_share",
        "zcta_county_assignment_quality",
        "dominant_county_ordinance_record",
        "dominant_county_ordinance_above_state",
        "any_county_ordinance_record",
        "any_county_ordinance_above_state",
    ]
    return feature[keep_columns].sort_values(["zcta5", "year"]).reset_index(drop=True)


def write_merged_listings_feature(feature: pd.DataFrame) -> None:
    listings_path = MODEL_READY_DIR / "listings_model_ready.csv"
    if not listings_path.exists():
        return

    listings = pd.read_csv(listings_path, dtype={"zcta5": str})
    listings["zcta5"] = listings["zcta5"].astype(str).str.zfill(5)
    listings["year"] = pd.to_numeric(listings["year"], errors="raise").astype(int)

    merged = listings.merge(
        feature,
        on=["zcta5", "year"],
        how="left",
        validate="many_to_one",
    )
    merged.to_csv(MERGED_LISTINGS_OUTPUT, index=False)

    missing = int(merged["minimum_wage_for_model_annual_avg"].isna().sum())
    print(f"Wrote merged listings preview: {MERGED_LISTINGS_OUTPUT}")
    print(f"Rows: {len(merged):,}")
    print(f"Rows missing minimum wage feature: {missing:,}")


def main() -> int:
    MODEL_READY_DIR.mkdir(parents=True, exist_ok=True)

    wage = load_minimum_wage()
    state = build_state_schedule(wage)
    clean = backfill_city_county_wage(wage, state)
    county_year = build_county_year_wage(clean, state)
    crosswalk = build_zcta_county_crosswalk()
    feature = build_zcta_features(crosswalk, county_year, state)

    clean.to_csv(CLEAN_OUTPUT, index=False)
    crosswalk.to_csv(CROSSWALK_OUTPUT, index=False)
    feature.to_csv(FEATURE_OUTPUT, index=False)

    print(f"Wrote cleaned city/county minimum wage table: {CLEAN_OUTPUT}")
    print(f"Rows: {len(clean):,}")
    print(f"Years: {clean['year'].min()}-{clean['year'].max()}")
    print(f"Backfilled rows: {int(clean['minimum_wage_backfilled'].sum()):,}")
    print()
    print(f"Wrote ZCTA-county crosswalk: {CROSSWALK_OUTPUT}")
    print(f"Rows: {len(crosswalk):,}")
    print(f"ZCTAs: {crosswalk['zcta5'].nunique():,}")
    print(
        "ZCTAs with low-dominance multi-county assignment: "
        f"{crosswalk.loc[crosswalk['is_dominant_county'] & crosswalk['zcta_county_assignment_quality'].eq('multi_county_low_dominance'), 'zcta5'].nunique():,}"
    )
    print()
    print(f"Wrote ZCTA-year minimum wage feature table: {FEATURE_OUTPUT}")
    print(f"Rows: {len(feature):,}")
    print(f"Years: {feature['year'].min()}-{feature['year'].max()}")
    print(
        "Rows using county ordinance above state: "
        f"{int(feature['dominant_county_ordinance_above_state'].sum()):,}"
    )
    print(
        "Rows with nonzero area-weighted difference from dominant county: "
        f"{int(feature['minimum_wage_area_weighted_diff_annual_avg'].abs().gt(1e-9).sum()):,}"
    )
    print()
    write_merged_listings_feature(feature)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
