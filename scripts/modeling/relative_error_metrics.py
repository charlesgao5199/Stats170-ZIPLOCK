#!/usr/bin/env python3
"""Add per-ZCTA relative error metrics to phase-2 holdout predictions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "phase2-models"
ACTUAL_PRICE_COLUMN = "y_next_year_median_sale_price"
PREDICTED_PRICE_COLUMN = "pred_next_year_median_sale_price"

MODEL_OUTPUTS = [
    {
        "model_label": None,
        "predictions_path": OUTPUT_ROOT / "linear-regression" / "linear_regression_2023_holdout_predictions.csv",
        "metrics_path": OUTPUT_ROOT / "linear-regression" / "linear_regression_2023_holdout_metrics.csv",
    },
    {
        "model_label": "random_forest",
        "predictions_path": OUTPUT_ROOT / "random-forest" / "random_forest_2023_holdout_predictions.csv",
        "metrics_path": OUTPUT_ROOT / "random-forest" / "random_forest_2023_holdout_metrics.csv",
    },
    {
        "model_label": "xgboost",
        "predictions_path": OUTPUT_ROOT / "xgboost" / "xgboost_2023_holdout_predictions.csv",
        "metrics_path": OUTPUT_ROOT / "xgboost" / "xgboost_2023_holdout_metrics.csv",
    },
]

CONSOLIDATED_OUTPUT = OUTPUT_ROOT / "phase2_holdout_relative_error_metrics.csv"


def add_relative_error_columns(
    frame: pd.DataFrame,
    actual_column: str = ACTUAL_PRICE_COLUMN,
    predicted_column: str = PREDICTED_PRICE_COLUMN,
) -> pd.DataFrame:
    """Return predictions with signed and absolute relative error columns.

    Relative error uses the teaching-team denominator: true median house price.
    """
    result = frame.copy()
    actual = pd.to_numeric(result[actual_column], errors="coerce")
    predicted = pd.to_numeric(result[predicted_column], errors="coerce")
    denominator = actual.where(actual.gt(0))

    result["relative_error"] = (predicted - actual) / denominator
    result["absolute_relative_error"] = result["relative_error"].abs()
    result["percentage_error"] = result["relative_error"] * 100.0
    result["absolute_percentage_error"] = result["absolute_relative_error"] * 100.0
    return result


def summarize_relative_errors(frame: pd.DataFrame) -> dict[str, float]:
    """Summarize already-computed relative errors as percent metrics."""
    valid = frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["relative_error", "absolute_relative_error"]
    )
    return {
        "rows": float(len(valid)),
        "mean_error_pct": float(valid["relative_error"].mean() * 100.0),
        "mape_pct": float(valid["absolute_relative_error"].mean() * 100.0),
        "median_ape_pct": float(valid["absolute_relative_error"].median() * 100.0),
        "rmse_pct": float(np.sqrt(np.mean(np.square(valid["relative_error"]))) * 100.0),
        "p90_ape_pct": float(valid["absolute_relative_error"].quantile(0.90) * 100.0),
        "max_ape_pct": float(valid["absolute_relative_error"].max() * 100.0),
    }


def summarize_by_model(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, group in predictions.groupby("model", sort=False):
        rows.append({"model": model_name, **summarize_relative_errors(group)})
    return pd.DataFrame(rows)


def update_metrics_file(metrics_path: Path, summary: pd.DataFrame) -> pd.DataFrame:
    metrics = pd.read_csv(metrics_path)
    relative_columns = [
        "rows",
        "mean_error_pct",
        "mape_pct",
        "median_ape_pct",
        "rmse_pct",
        "p90_ape_pct",
        "max_ape_pct",
    ]
    metrics = metrics.drop(columns=[col for col in relative_columns if col in metrics.columns])

    if "model" in metrics.columns:
        updated = metrics.merge(summary, on="model", how="left", validate="one_to_one")
    else:
        updated = pd.concat(
            [
                summary[["model"]].reset_index(drop=True),
                metrics.reset_index(drop=True),
                summary.drop(columns=["model"]).reset_index(drop=True),
            ],
            axis=1,
        )
    updated.to_csv(metrics_path, index=False)
    return updated


def update_model_output(spec: dict[str, object]) -> pd.DataFrame:
    predictions_path = Path(spec["predictions_path"])
    metrics_path = Path(spec["metrics_path"])
    model_label = spec["model_label"]

    predictions = pd.read_csv(predictions_path, dtype={"zcta5": str})
    if "model" not in predictions.columns:
        predictions["model"] = str(model_label)

    predictions = add_relative_error_columns(predictions)
    predictions.to_csv(predictions_path, index=False)

    summary = summarize_by_model(predictions)
    updated_metrics = update_metrics_file(metrics_path, summary)

    metadata_columns = [
        col
        for col in ["model", "train_years", "test_year", "target_year", "ridge_alpha"]
        if col in updated_metrics.columns
    ]
    metric_columns = [
        col
        for col in [
            "mae_dollars",
            "rmse_dollars",
            "r2_dollars",
            "mae_log",
            "rmse_log",
            "r2_log",
            "rows",
            "mean_error_pct",
            "mape_pct",
            "median_ape_pct",
            "rmse_pct",
            "p90_ape_pct",
            "max_ape_pct",
        ]
        if col in updated_metrics.columns
    ]
    return updated_metrics[metadata_columns + metric_columns]


def main() -> int:
    summaries = [update_model_output(spec) for spec in MODEL_OUTPUTS]
    consolidated = pd.concat(summaries, ignore_index=True, sort=False)
    consolidated.to_csv(CONSOLIDATED_OUTPUT, index=False)

    display_columns = [
        col
        for col in ["model", "mape_pct", "median_ape_pct", "rmse_pct", "r2_log", "r2_dollars"]
        if col in consolidated.columns
    ]
    print(f"Wrote consolidated relative-error metrics: {CONSOLIDATED_OUTPUT}")
    print(consolidated[display_columns].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
