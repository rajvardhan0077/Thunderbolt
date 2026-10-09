# ML Methodology

## Problem framing

This project solves three main prediction tasks:

- availability classification
- wait-time regression
- price regression

## Dataset

The dataset contains synthetic/demo records for Royal Enfield Thunderbird 350X spare parts. Columns include:

- Bike model
- Manufacturing year
- Part name
- Part category
- Availability status
- Estimated wait days
- Estimated price (INR)
- Supplier type
- City
- Garage type
- Part condition
- Service priority
- Record date

## Preprocessing

The preprocessing pipeline includes:

- duplicate check
- missing-value handling
- date feature extraction
- one-hot encoding for categorical variables
- numeric imputation for continuous columns
- train/test split with a fixed random seed

## Algorithms used

### Availability classification

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

### Waiting time regression

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

### Price regression

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

## Evaluation metrics

- Accuracy
- Weighted precision
- Weighted recall
- Weighted F1-score
- MAE
- MSE
- RMSE
- R²

## Model saving

The best model for each task is saved as:

- `ML/models/availability_model.pkl`
- `ML/models/wait_model.pkl`
- `ML/models/price_model.pkl`
- `ML/models/metrics.json`

## Important note

The score is only a prototype estimate. It should not be interpreted as live market or stock data.
