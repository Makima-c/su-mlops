# Changelog

## v0.2

I switched from LinearRegression to RandomForestRegressor to try capturing
nonlinear relationships between the features and progression score.
The API still takes the same 10 features and returns a numeric prediction.

I kept the same 80/20 split and seed 42, and chose the forest parameters using
5-fold cross-validation on the training set: 200 trees, no depth limit,
and at least 5 samples per leaf.

| Version | Model | Test RMSE |
| --- | --- | ---: |
| v0.1 | StandardScaler + LinearRegression | 53.8534 |
| v0.2 | RandomForestRegressor | 53.5283 |

RMSE dropped by 0.3252, about 0.60%. The gain is small and was measured on this
split. The forest model is also larger than the baseline.

## v0.1

Started with StandardScaler + LinearRegression on the sklearn Diabetes dataset.
With an 80/20 split and seed 42, there were 353 training rows and 89 test rows.
Test RMSE was 53.8534.

Added the Flask API, JSON input errors and a Docker image with the model included.
