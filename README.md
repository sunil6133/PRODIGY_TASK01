# PRODIGY_ML_TASK01

House Price Prediction using Linear Regression

This repository contains a Python script to implement a linear regression model to predict house prices based on square footage (number of rooms) and the number of bedrooms. The dataset used for this project is a subset of the Boston Housing dataset.

Dataset: https://www.kaggle.com/datasets/vikrishnan/boston-house-prices
The dataset includes the following columns:

CRIM: per capita crime rate by town
ZN: proportion of residential land zoned for lots over 25,000 sq. ft.
INDUS: proportion of non-retail business acres per town
CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
NOX: nitric oxides concentration (parts per 10 million)
RM: average number of rooms per dwelling
AGE: proportion of owner-occupied units built prior to 1940
DIS: weighted distances to five Boston employment centers
RAD: index of accessibility to radial highways
TAX: full-value property tax rate per $10,000
PTRATIO: pupil-teacher ratio by town
B: 1000(Bk - 0.63)^2 where Bk is the proportion of black residents by town
LSTAT: % lower status of the population
MEDV: Median value of owner-occupied homes in $1000s
In this implementation, we focus on the following features:

RM: average number of rooms per dwelling
RAD: index of accessibility to radial highways (used as a proxy for the number of bedrooms)
The target variable is:

MEDV: Median value of owner-occupied homes in $1000s
Project Structure
house_price_prediction.py: Main script to load data, train the linear regression model, and make predictions.
README.md: This file, providing an overview of the project.
Requirements

To run this project, you need the following Python packages:
numpy
pandas
scikit-learn

You can install the required packages using pip:
pip install numpy pandas scikit-learn


Clone this repository:
git clone https://github.com/SaiSamhitha06/house-price-prediction.git
cd house-price-prediction

Run the script:
The script will:
Load the dataset into a pandas DataFrame.
Select the relevant features (RM and RAD) and target variable (MEDV).
Split the dataset into training and testing sets.
Train a linear regression model.
Evaluate the model using Mean Squared Error (MSE) and R-squared metrics.
Print the model's intercept and coefficients.
Make a sample prediction using the trained model.


Example Output:
Mean Squared Error: 31.78
R-squared: 0.64
Intercept: 12.53
Coefficients: [ 8.43 -0.12]
Predicted price for sample data: 64.77
