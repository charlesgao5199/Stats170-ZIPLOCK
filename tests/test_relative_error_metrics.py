from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "modeling"))

from relative_error_metrics import (
    add_relative_error_columns,
    summarize_relative_errors,
    update_metrics_file,
)


def test_add_relative_error_columns_uses_true_price_as_denominator():
    frame = pd.DataFrame(
        {
            "actual": [100.0, 200.0],
            "predicted": [110.0, 150.0],
        }
    )

    result = add_relative_error_columns(frame, "actual", "predicted")

    assert result["relative_error"].tolist() == [0.10, -0.25]
    assert result["percentage_error"].tolist() == [10.0, -25.0]
    assert result["absolute_percentage_error"].tolist() == [10.0, 25.0]


def test_summarize_relative_errors_reports_per_row_percent_metrics():
    frame = pd.DataFrame(
        {
            "relative_error": [0.10, -0.25],
            "absolute_relative_error": [0.10, 0.25],
            "percentage_error": [10.0, -25.0],
            "absolute_percentage_error": [10.0, 25.0],
        }
    )

    summary = summarize_relative_errors(frame)

    assert summary["mean_error_pct"] == -7.5
    assert summary["mape_pct"] == 17.5
    assert summary["median_ape_pct"] == 17.5
    assert summary["rmse_pct"] == np.sqrt((0.10**2 + 0.25**2) / 2) * 100.0


def test_update_metrics_file_adds_model_label_when_metrics_file_has_no_model_column(tmp_path):
    metrics_path = tmp_path / "metrics.csv"
    pd.DataFrame(
        [{"mae_dollars": 10.0, "rmse_dollars": 12.0, "r2_log_change": 0.9}]
    ).to_csv(metrics_path, index=False)
    summary = pd.DataFrame(
        [
            {
                "model": "random_forest",
                "rows": 2.0,
                "mean_error_pct": -7.5,
                "mape_pct": 17.5,
                "median_ape_pct": 17.5,
                "rmse_pct": 19.0,
                "p90_ape_pct": 23.5,
                "max_ape_pct": 25.0,
            }
        ]
    )

    updated = update_metrics_file(metrics_path, summary)

    assert updated.loc[0, "model"] == "random_forest"
    assert updated.loc[0, "mape_pct"] == 17.5
