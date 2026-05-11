#!/usr/bin/env python3
"""Build annual ZCTA-year listings features and next-year targets for modeling."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "model-datasets" / "listings"
DEMOGRAPHICS_PATH = PROJECT_ROOT / "model-datasets" / "zcta_demographics"
OUTPUT_DIR = PROJECT_ROOT / "model-ready"
ANNUAL_OUTPUT = OUTPUT_DIR / "listings_annual_all_residential.csv"
MODEL_OUTPUT = OUTPUT_DIR / "listings_model_ready.csv"

PROPERTY_TYPE = "All Residential"

INPUT_COLUMNS = [
    "REGION",
    "YEAR",
    "PROPERTY_TYPE",
    "PARENT_METRO_REGION",
    "PERIOD_END",
    "MEDIAN_SALE_PRICE",
    "MEDIAN_LIST_PRICE",
    "MEDIAN_PPSF",
    "MEDIAN_LIST_PPSF",
    "HOMES_SOLD",
    "PENDING_SALES",
    "NEW_LISTINGS",
    "INVENTORY",
    "OFF_MARKET_IN_TWO_WEEKS",
    "AVG_SALE_TO_LIST",
    "SOLD_ABOVE_LIST",
    "MEDIAN_DOM",
]

NUMERIC_COLUMNS = [
    "MEDIAN_SALE_PRICE",
    "MEDIAN_LIST_PRICE",
    "MEDIAN_PPSF",
    "MEDIAN_LIST_PPSF",
    "HOMES_SOLD",
    "PENDING_SALES",
    "NEW_LISTINGS",
    "INVENTORY",
    "OFF_MARKET_IN_TWO_WEEKS",
    "AVG_SALE_TO_LIST",
    "SOLD_ABOVE_LIST",
    "MEDIAN_DOM",
]


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """Compute a weighted mean and fall back to an unweighted mean if needed."""
    valid = values.notna() & weights.notna() & (weights > 0)
    if valid.any():
        return float(np.average(values[valid], weights=weights[valid]))

    values_only = values.dropna()
    if not values_only.empty:
        return float(values_only.mean())

    return float("nan")


def first_nonnull(series: pd.Series) -> str | None:
    nonnull = series.dropna()
    if nonnull.empty:
        return None
    return str(nonnull.iloc[0])


def summarize_group(group: pd.DataFrame) -> dict[str, object]:
    group = group.sort_values("period_end").copy()
    months = group["month"].dropna().astype(int)

    annual_homes_sold = group["HOMES_SOLD"].sum(min_count=1)
    annual_pending_sales = group["PENDING_SALES"].sum(min_count=1)
    annual_new_listings = group["NEW_LISTINGS"].sum(min_count=1)
    annual_inventory_mean = group["INVENTORY"].mean()
    annual_inventory_end = group["INVENTORY"].dropna().iloc[-1] if group["INVENTORY"].notna().any() else np.nan

    annual_median_sale_price = weighted_mean(group["MEDIAN_SALE_PRICE"], group["HOMES_SOLD"])
    annual_median_list_price = weighted_mean(group["MEDIAN_LIST_PRICE"], group["NEW_LISTINGS"])
    annual_median_ppsf = weighted_mean(group["MEDIAN_PPSF"], group["HOMES_SOLD"])
    annual_median_list_ppsf = weighted_mean(group["MEDIAN_LIST_PPSF"], group["NEW_LISTINGS"])
    annual_avg_sale_to_list = weighted_mean(group["AVG_SALE_TO_LIST"], group["HOMES_SOLD"])
    annual_sold_above_list = weighted_mean(group["SOLD_ABOVE_LIST"], group["HOMES_SOLD"])
    annual_off_market_in_two_weeks = weighted_mean(group["OFF_MARKET_IN_TWO_WEEKS"], group["NEW_LISTINGS"])
    annual_median_dom = weighted_mean(group["MEDIAN_DOM"], group["HOMES_SOLD"])

    annual_months_of_supply_est = np.nan
    if pd.notna(annual_inventory_mean) and pd.notna(annual_homes_sold) and annual_homes_sold > 0:
        annual_months_of_supply_est = float(annual_inventory_mean / (annual_homes_sold / 12.0))

    return {
        "zcta5": str(group["zcta5"].iloc[0]).zfill(5),
        "year": int(group["year"].iloc[0]),
        "parent_metro_region": first_nonnull(group["PARENT_METRO_REGION"]),
        "months_observed": int(months.nunique()),
        "first_month": int(months.min()) if not months.empty else np.nan,
        "last_month": int(months.max()) if not months.empty else np.nan,
        "full_year_coverage": bool(months.nunique() == 12),
        "months_with_median_sale_price": int(group["MEDIAN_SALE_PRICE"].notna().sum()),
        "months_with_median_list_price": int(group["MEDIAN_LIST_PRICE"].notna().sum()),
        "months_with_homes_sold": int(group["HOMES_SOLD"].notna().sum()),
        "months_with_pending_sales": int(group["PENDING_SALES"].notna().sum()),
        "months_with_new_listings": int(group["NEW_LISTINGS"].notna().sum()),
        "months_with_inventory": int(group["INVENTORY"].notna().sum()),
        "annual_homes_sold": float(annual_homes_sold) if pd.notna(annual_homes_sold) else np.nan,
        "annual_pending_sales": float(annual_pending_sales) if pd.notna(annual_pending_sales) else np.nan,
        "annual_new_listings": float(annual_new_listings) if pd.notna(annual_new_listings) else np.nan,
        "annual_inventory_mean": float(annual_inventory_mean) if pd.notna(annual_inventory_mean) else np.nan,
        "annual_inventory_end": float(annual_inventory_end) if pd.notna(annual_inventory_end) else np.nan,
        "annual_months_of_supply_est": annual_months_of_supply_est,
        "annual_median_sale_price": annual_median_sale_price,
        "annual_median_list_price": annual_median_list_price,
        "annual_median_ppsf": annual_median_ppsf,
        "annual_median_list_ppsf": annual_median_list_ppsf,
        "annual_avg_sale_to_list": annual_avg_sale_to_list,
        "annual_sold_above_list": annual_sold_above_list,
        "annual_off_market_in_two_weeks": annual_off_market_in_two_weeks,
        "annual_median_dom": annual_median_dom,
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    valid_zctas = set(
        pd.read_csv(DEMOGRAPHICS_PATH, usecols=["zcta5"], dtype=str)["zcta5"]
        .dropna()
        .astype(str)
        .str.zfill(5)
        .unique()
        .tolist()
    )

    listings = pd.read_csv(INPUT_PATH, usecols=INPUT_COLUMNS, dtype=str, low_memory=False)
    listings = listings[listings["PROPERTY_TYPE"] == PROPERTY_TYPE].copy()
    listings["zcta5"] = listings["REGION"].astype(str).str.strip().str.zfill(5)
    listings = listings[listings["zcta5"].isin(valid_zctas)].copy()
    listings["year"] = pd.to_numeric(listings["YEAR"], errors="coerce")
    listings["period_end"] = pd.to_datetime(listings["PERIOD_END"], utc=True, errors="coerce")
    listings["month"] = listings["period_end"].dt.month

    for col in NUMERIC_COLUMNS:
        listings[col] = pd.to_numeric(listings[col], errors="coerce")

    listings = listings.dropna(subset=["year", "period_end"]).copy()
    listings["year"] = listings["year"].astype(int)

    duplicate_count = int(
        listings.duplicated(subset=["zcta5", "year", "period_end"]).sum()
    )
    if duplicate_count:
        raise ValueError(f"Found {duplicate_count} duplicate zcta5-year-period rows after filtering.")

    records: list[dict[str, object]] = []
    for _, group in listings.groupby(["zcta5", "year"], sort=True):
        records.append(summarize_group(group))

    annual = pd.DataFrame(records).sort_values(["zcta5", "year"]).reset_index(drop=True)

    next_year = annual[
        ["zcta5", "year", "months_observed", "full_year_coverage", "annual_median_sale_price"]
    ].copy()
    next_year["year"] = next_year["year"] - 1
    next_year = next_year.rename(
        columns={
            "months_observed": "next_year_months_observed",
            "full_year_coverage": "next_year_full_year_coverage",
            "annual_median_sale_price": "next_year_annual_median_sale_price",
        }
    )

    model = annual.merge(next_year, on=["zcta5", "year"], how="left")
    max_feature_year = int(annual["year"].max()) - 1
    model = model[model["year"] <= max_feature_year].copy()

    current_price = model["annual_median_sale_price"]
    next_price = model["next_year_annual_median_sale_price"]

    model["target_median_sale_price_pct_change_next_year"] = np.where(
        current_price.gt(0) & next_price.gt(0),
        (next_price / current_price) - 1.0,
        np.nan,
    )
    model["target_median_sale_price_log_change_next_year"] = np.where(
        current_price.gt(0) & next_price.gt(0),
        np.log(next_price) - np.log(current_price),
        np.nan,
    )
    model["has_next_year_target"] = model["target_median_sale_price_pct_change_next_year"].notna()
    model["target_next_year_complete"] = model["next_year_full_year_coverage"].eq(True)
    model["baseline_model_eligible"] = (
        model["full_year_coverage"] & model["has_next_year_target"] & model["target_next_year_complete"]
    )

    annual.to_csv(ANNUAL_OUTPUT, index=False)
    model.to_csv(MODEL_OUTPUT, index=False)

    print(f"Wrote annual listings table: {ANNUAL_OUTPUT}")
    print(f"Rows: {len(annual):,}")
    print(f"Columns: {len(annual.columns):,}")
    print()
    print(f"Wrote listings modeling table: {MODEL_OUTPUT}")
    print(f"Rows: {len(model):,}")
    print(f"Columns: {len(model.columns):,}")
    print(f"Rows with next-year target: {int(model['has_next_year_target'].sum()):,}")
    print(f"Rows with complete next-year target: {int(model['target_next_year_complete'].sum()):,}")
    print(f"Baseline-eligible rows: {int(model['baseline_model_eligible'].sum()):,}")
    print(f"Valid California ZCTAs retained: {annual['zcta5'].nunique():,}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
