# Credit Card Fraud Detection

Machine learning model to detect fraudulent credit card transactions.

## Problem
Fraudulent transactions are rare, so the data is highly imbalanced. This project builds and compares models that catch fraud while keeping false alarms low.

## Dataset
Kaggle "Credit Card Fraud Detection" dataset (284,807 transactions, about 0.17% fraud). Download it and place `creditcard.csv` in the `data/` folder. It is not included in this repo because of its size.

## Project Structure
- `data/` - dataset (not tracked)
- `notebooks/` - EDA and modeling notebooks
- `src/` - reusable Python scripts
- `requirements.txt` - dependencies

## Approach
1. Exploratory data analysis
2. Handling class imbalance (SMOTE / class weights)
3. Training models: Logistic Regression, Random Forest, XGBoost
4. Evaluation: Precision, Recall, F1, PR-AUC

## Results
To be added.

## How to Run
pip install -r requirements.txt
jupyter notebook

## Author
Areebah Mahmood
Sania Taqui
