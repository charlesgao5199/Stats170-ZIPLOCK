# Expo-Ready Poster Assets

This folder contains high-resolution figures and summary tables for the project expo poster.

## Core Graphs

- `feature_graph_ridge_top20_coefficients.png`: interpretable Ridge coefficient graph.
- `feature_graph_random_forest_top20_permutation_importance.png`: Random Forest feature importance graph.
- `feature_graph_xgboost_top20_permutation_importance.png`: XGBoost feature importance graph.
- `predicted_vs_actual_ridge_2023_holdout.png`: Ridge predicted vs. actual 2024 prices.
- `predicted_vs_actual_random_forest_2023_holdout.png`: Random Forest predicted vs. actual 2024 prices.
- `predicted_vs_actual_xgboost_2023_holdout.png`: XGBoost predicted vs. actual 2024 prices.

## Extra Poster Assets

- `model_holdout_summary_for_poster.png`: compact model-performance table for the poster.
- `model_holdout_summary_for_poster.csv`: editable version of the poster table.
- `model_holdout_relative_error_metrics.csv`: full holdout metrics with percent relative error.
- `model_example_random_forest_tree_depth1.png`: simple model-logic visual if you want an explainability panel.

## Suggested Poster Content

- State the target clearly: predict `log(2024 median sale price / 2023 median sale price)`, then reconstruct 2024 dollars from 2023 price.
- Use percent relative error as the main accuracy metric, since it compares ZIP codes fairly across different price levels.
- Include one predicted-vs-actual graph and one feature-importance graph to avoid overcrowding.
- Mention the negative `R2 log change` directly: price growth is harder to predict than price level, while dollar-price `R2` is high because current price anchors next-year price.
- Add a short limitations box: macroeconomic factors such as interest rates, mortgage rates, inflation, and unemployment are not directly included.
