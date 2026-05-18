from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "modeling"))

import prepare_phase2_model_data as prep


def write_minimal_phase2_table(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "zcta5": "90001",
                "year": 2023,
                "annual_median_sale_price": 100.0,
                "next_year_annual_median_sale_price": 110.0,
                "target_median_sale_price_pct_change_next_year": 0.10,
                "target_median_sale_price_log_change_next_year": np.log(1.10),
                "target_next_year_complete": True,
            }
        ]
    ).to_csv(path, index=False)


def test_load_phase2_table_uses_log_change_as_model_target(tmp_path):
    input_path = tmp_path / "phase2.csv"
    write_minimal_phase2_table(input_path)

    frame = prep.load_phase2_table(input_path)

    assert prep.TARGET_ALIAS == "y_next_year_median_sale_price_log_change"
    assert prep.PRICE_TARGET_ALIAS == "y_next_year_median_sale_price"
    assert np.isclose(frame.loc[0, prep.TARGET_ALIAS], np.log(1.10))
    assert frame.loc[0, prep.PRICE_TARGET_ALIAS] == 110.0


def test_choose_feature_columns_excludes_targets_but_keeps_current_price(tmp_path):
    input_path = tmp_path / "phase2.csv"
    write_minimal_phase2_table(input_path)
    frame = prep.load_phase2_table(input_path)

    features = prep.choose_feature_columns(frame)

    assert "annual_median_sale_price" in features
    assert prep.TARGET_ALIAS not in features
    assert prep.PRICE_TARGET_ALIAS not in features
    assert "target_median_sale_price_log_change_next_year" not in features
    assert "next_year_annual_median_sale_price" not in features
