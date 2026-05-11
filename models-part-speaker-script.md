# Models Part Speaker Script

## Slide 1: Modeling Goal and Setup

For the machine learning section, our goal is to predict next-year median sale price at the ZCTA-year level. Each row represents one California ZCTA in one year. We trained and cross-validated using observations from 2012 through 2022, then held out 2023 as the final test set. Because the target for each row is next-year median sale price, the 2023 test rows are evaluated against 2024 median sale price.

We used the same 54 selected features for all models so the comparison is fair. These features include prior housing-market variables, demographics, school quality, crime, amenities, and minimum wage. We modeled log median sale price because housing prices are highly skewed, but we report performance both in log scale and dollars.

## Slide 2: Linear Regression

The first model is linear regression, specifically Ridge regression as the main linear version. We included this model because it is simple and interpretable. It is a useful benchmark: if a more complex model cannot beat this, then the added complexity is probably not justified.

Linear regression estimates one coefficient for each feature, which tells us how the feature is associated with log home price while holding other variables constant. We used Ridge regularization, which adds an L2 penalty. This helps reduce coefficient instability when features are correlated, which is common in housing data.

One important categorical feature here is parent_metro_region. This captures the broader metro housing market that each ZCTA belongs to, such as Los Angeles, San Francisco, or Sacramento. That matters because housing prices can differ a lot across regional markets.

## Slide 3: Linear Regression Continued

This predicted-versus-actual plot shows how Ridge performs on the 2023 holdout, which predicts 2024 prices. The left panel uses a log scale so we can see the full range, including high-price ZCTAs. The right panel zooms into the main cluster.

The linear model has reasonable log-scale performance, but it struggles badly on the dollar scale. The main issue is extrapolation for expensive ZCTAs. A few high-end areas are overpredicted by a large amount, which makes the dollar RMSE very large and gives a negative dollar R-squared. So this model is useful for interpretation, but not our best predictive model.

## Slide 4: Random Forest

The second model is Random Forest. We used it because it can capture nonlinear relationships and interactions without requiring us to specify them manually. It is also useful for feature importance.

A Random Forest is made of many decision trees. Each tree repeatedly splits the data based on feature thresholds. For example, a tree might first split on household income, then split again on race share, education share, sale price, or another feature. Each individual tree is noisy, but the forest averages predictions across many trees, which makes the final prediction more stable.

The tree visual here is not the full forest. It is just one example tree, truncated to the first two levels so we can show the inner working without making the slide unreadable.

## Slide 5: Random Forest Continued

This slide shows the Random Forest holdout predictions. Compared with the linear model, the point cloud is closer to the diagonal line, which means predicted prices are closer to actual prices.

The Random Forest performs much better than Ridge on the final holdout. Its MAE is about $103K, RMSE is about $246K, and dollar R-squared is about 0.874. The model still tends to underpredict some very expensive ZCTAs, but it avoids the extreme high-end overprediction problem we saw in the linear model.

## Slide 6: XGBoost

The third model is XGBoost. We used it because it is designed to capture nonlinear relationships and feature interactions while controlling overfitting.

XGBoost is a gradient boosting method. Instead of building independent trees like Random Forest, it builds trees sequentially. Each new tree tries to correct the errors made by the previous trees. We used many shallow trees with a small learning rate, plus row and feature subsampling and L2 regularization. In our setup, the key parameters were 700 trees, learning rate 0.035, max depth 4, subsample 0.85, colsample by tree 0.85, and L2 regularization lambda 5.

The feature importance plot shows that annual median sale price dominates. This makes sense because the best predictor of next year's home price is the current local housing price level. Other important variables include price per square foot, education share, household income, and recent market behavior.

## Slide 7: XGBoost Continued

This predicted-versus-actual plot shows that XGBoost is the strongest model. Most points are very close to the diagonal line, especially in the main cluster. The model still misses some high-end ZCTAs, but the errors are much smaller than with the other two methods.

On the 2023 holdout, XGBoost has an MAE of about $68.8K, RMSE of about $135.1K, and dollar R-squared of about 0.962. It also has the best log-scale R-squared, about 0.975.

## Slide 8: Results

This table summarizes the main comparison. Ridge regression is the most interpretable, but it performs worst for prediction because the linear structure does not handle the high-end price distribution well. Random Forest improves substantially because it captures nonlinear patterns and uses many decision trees. XGBoost performs best overall because boosting can learn more refined nonlinear corrections.

The main conclusion is that XGBoost is our strongest predictive model, while Random Forest is useful for interpreting feature importance and Linear Regression is useful as a transparent baseline. Across the models, the most important predictors are prior housing-market features, especially annual median sale price and annual median price per square foot, followed by socioeconomic variables like income and education.

## Rubric Coverage Check

Description of methods used: Covered. The presentation explains Linear/Ridge Regression, Random Forest, and XGBoost, including how each model works at a high level.

Justification of methods: Covered. Linear regression is the interpretable benchmark, Random Forest captures nonlinear patterns and feature importance, and XGBoost is the flexible boosted-tree model for strongest prediction.

Checking assumptions and diagnostics: Partially covered. The current slides imply this through predicted-versus-actual plots and the discussion of linear regression extrapolation, but this rubric item should be made more explicit in the spoken script or with one extra slide. Mention that linear regression assumes a stable linear relationship in log price, while tree models require fewer linearity assumptions but are checked through time-blocked cross-validation and holdout performance.

Preliminary results and visual presentation: Covered. The deck includes coefficient or importance plots, predicted-versus-actual plots, and a final metrics table.

Interpretation and how to tell if results are good: Covered, but should be stated clearly on the results slide. A good result has low MAE/RMSE, high R-squared, strong holdout performance, and predicted-versus-actual points close to the diagonal line. XGBoost is best by these criteria.

## Suggested Rubric Fixes

Add one sentence on Slide 1 or Slide 8: "We used expanding time-blocked cross-validation and a final 2023 holdout test to check generalization."

Add one sentence on Slide 3: "The linear model's diagnostic issue is high-end extrapolation, visible in the predicted-versus-actual plot."

Add one sentence on Slide 8: "We define a good model as one with lower MAE/RMSE, higher holdout R-squared, and predictions close to the diagonal."

Fix small slide text issues: change "Regularization, shrinkage, and subsampling.." to "Regularization, shrinkage, and subsampling." Also consider writing "Median sale price dominates" instead of "Median_sale_price dominates !!!" for a more presentation-ready tone.
