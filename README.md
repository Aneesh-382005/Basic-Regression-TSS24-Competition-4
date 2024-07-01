# TSS24 Competition 4: Regression Task on Kaggle

This repository contains my submission for the TSS24 Competition 4 on Kaggle. The task involves predicting the number of rings in abalone datasets using various regression models and ensemble methods.

## Project Overview

The goal of this project is to predict the target variable `Rings` using features from the provided dataset. We utilize various regression models and ensemble techniques to improve the prediction accuracy.

## Model Training and Evaluation

The models are trained using the following steps:

1. Splitting the dataset into training and test sets.
2. Training various regression models on the training set.
3. Evaluating model performance using the RMSLE (Root Mean Squared Logarithmic Error) metric.

## Ensemble Method

To improve the prediction accuracy, we used an ensemble method by averaging the predictions from the XGBoost and Random Forest models.

## Hyperparameter Tuning

Manual tuning of hyperparameters to address overfitting issues:

```python
modelrf = RandomForestRegressor(
    n_estimators = 100, 
    max_depth = 10, 
    min_samples_split = 10, 
    min_samples_leaf = 4, 
    max_features = 'sqrt', 
    random_state = 42
)
