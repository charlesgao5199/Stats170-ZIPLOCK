# Stats 170B Project

Phase 2 should treat the repository as a small data pipeline with separate areas for inputs, modeling-ready tables, code, notebooks, archived intermediates, and generated outputs.

## Folder Layout

- `model-datasets/`: main comprehensive input datasets for phase 2.
- `model-ready/`: derived CSV tables used directly by modeling notebooks and report code.
- `scripts/pipeline/`: current data-prep and report-figure scripts.
- `scripts/zcta_archive/`: older ACS/ZCTA collection, validation, SQLite, and exploratory plotting utilities.
- `notebooks/modeling/`: modeling and report notebooks.
- `phase-2-models/`: phase 2 model notebooks for Random Forest, Linear Regression, and XGBoost.
- `notebooks/zcta_archive/`: older ZCTA exploratory notebooks.
- `data/zcta_archive/`: archived raw ACS/ZCTA source files and intermediate tables from phase 1.
- `outputs/report-figures/`: generated report figures and metrics.
- `outputs/phase2-models/`: phase 2 model-ready splits, metrics, predictions, and feature-importance outputs.
- `outputs/zcta-exploration/`: generated exploratory ZCTA plots.
- `outputs/cache/`: generated runtime/cache artifacts.

## Common Commands

Run these from the project root:

```bash
python3 scripts/pipeline/build_demographics_feature_table.py
python3 scripts/pipeline/build_listings_model_table.py
python3 scripts/pipeline/build_school_quality_feature_table.py
python3 scripts/pipeline/build_minimum_wage_feature_table.py
python3 scripts/pipeline/build_crime_feature_table.py
python3 scripts/pipeline/build_amenities_feature_table.py
python3 scripts/pipeline/build_phase2_model_table.py
python3 scripts/pipeline/build_report_figures.py
python3 scripts/modeling/prepare_phase2_model_data.py
python3 scripts/modeling/select_phase2_features.py
python3 scripts/modeling/relative_error_metrics.py
```

The scripts resolve paths from the project root, so they can also be invoked by absolute path.

Phase 2 model notebooks:

```bash
jupyter nbconvert --to notebook --execute --inplace phase-2-models/random_forest_phase2.ipynb
jupyter nbconvert --to notebook --execute --inplace phase-2-models/linear_regression_phase2.ipynb
jupyter nbconvert --to notebook --execute --inplace phase-2-models/xgboost_phase2.ipynb
```
