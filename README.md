# Stats 170B Final Project Submission

Project title: California Home Price Forecasting

Team members:
- Gary Zeng
- Charles Gao
- Anuj Patel
- Allen Lai

This folder is the compact final GitHub submission package for the Stats 170B project. It contains a runnable demonstration notebook, the small data files needed by that notebook, selected project scripts, and the final report PDF.

## How to run

From this folder:

```bash
python3 -m pip install -r requirements.txt
jupyter nbconvert --to notebook --execute project.ipynb --output project.executed.ipynb
jupyter nbconvert --to html project.executed.ipynb --output project.html
```

The notebook is designed to run in under one minute on the included compact data. It trains a Ridge regression model on the selected 2012-2022 training data, evaluates it on the 2023 holdout that predicts 2024 prices, and displays the saved final comparison metrics for Ridge, Random Forest, and XGBoost.

## Files

- `README.md`: this file; describes the submission contents and how to run the project demo.
- `requirements.txt`: Python package requirements for the runnable notebook.
- `project.ipynb`: runnable final project demonstration notebook.
- `project.html`: rendered HTML version of the notebook with cell outputs.
- `data/phase2_selected_train_2012_2022.csv`: selected 54-feature training data used by the demo notebook.
- `data/phase2_selected_test_2023.csv`: selected 2023 holdout data used to evaluate 2024 price forecasts.
- `data/phase2_selected_features.json`: feature schema and preprocessing groups for the selected model features.
- `data/phase2_selected_feature_schema.csv`: tabular schema for the selected features.
- `data/phase2_holdout_relative_error_metrics.csv`: saved final holdout metrics for Ridge, Random Forest, and XGBoost.
- `outputs/linear_regression_coefficients_by_feature.csv`: Ridge feature coefficient summary from the final model run.
- `outputs/random_forest_permutation_importance.csv`: Random Forest permutation-importance output.
- `outputs/xgboost_permutation_importance.csv`: XGBoost permutation-importance output.
- `scripts/modeling/`: modeling scripts used to prepare selected features and relative-error metrics in the full project.
- `scripts/pipeline/`: pipeline scripts used to construct feature tables in the full project.
- `report/final_report.pdf`: final project report PDF.

## Data-size note

The original working directory contains raw and intermediate data that are much larger than the course upload limit. This submission folder includes only the compact selected train/test data and final metrics needed to run the demonstration notebook. The files under `data/` are under the 20 MB course limit.

## Project summary

The project predicts next-year median sale prices for California ZIP Code Tabulation Areas (ZCTAs). The final modeling table combines housing, demographic, school, crime, amenity, and minimum-wage features. The final evaluation uses 2023 features to predict 2024 median sale prices and compares Ridge regression, Random Forest, and XGBoost. Ridge has the lowest MAPE on the final holdout, while Random Forest has the highest dollar-scale R-squared.
