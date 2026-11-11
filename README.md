# Credit Card Fraud Detection

Machine learning models to detect fraudulent credit card transactions, with a focus on handling extreme class imbalance.

## Problem
Fraud is rare: only 0.17% of transactions in this dataset are fraudulent. A model that always predicts "normal" would be 99.8% accurate and catch no fraud, so this project compares models using recall, precision, F1-score and PR-AUC instead of accuracy.

## Dataset
Kaggle "Credit Card Fraud Detection" (ULB): 284,807 transactions by European cardholders over two days, 492 of them fraud. Features are Time, Amount, and V1 to V28 (anonymized PCA components). Download `creditcard.csv` and place it in the `data/` folder. It is not included in this repo because of its size.

## Project Structure
- `notebooks/01_EDA.ipynb`: exploratory data analysis
- `notebooks/02_Preprocessing.ipynb`: duplicate removal, train/test split, scaling
- `notebooks/03_Modelling.ipynb`: models, imbalance handling and evaluation
- `data/`: dataset (not tracked)
- `requirements.txt`: dependencies

## Approach
1. Exploratory data analysis: no missing values, 1,081 duplicate rows removed, severe class imbalance
2. Stratified 80/20 train/test split; Time and Amount scaled (scaler fitted on the training set only)
3. Imbalance handling: class weights and SMOTE (applied to the training set only)
4. Models: Logistic Regression, Decision Tree, Random Forest, XGBoost
5. Evaluation on the untouched test set (56,746 transactions, 95 fraud)

## Key Findings from EDA
- Fraud is 0.17% of transactions (492 of 284,807)
- Fraud is more frequent in the early hours of the day (peak around hour 2)
- V17, V14, V12 and V10 are the strongest fraud signals; fraud values are shifted towards negative
- Amount alone cannot separate fraud from normal transactions

## Results

| Model | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| XGBoost | 0.738 | 0.800 | 0.768 | 0.804 |
| Random Forest | 0.971 | 0.705 | 0.817 | 0.798 |
| Logistic Regression (baseline) | 0.848 | 0.589 | 0.696 | 0.692 |
| Logistic Regression (SMOTE) | 0.053 | 0.874 | 0.100 | 0.677 |
| Logistic Regression (class weights) | 0.056 | 0.874 | 0.106 | 0.672 |
| Decision Tree | 0.068 | 0.832 | 0.126 | 0.470 |

**Best model: XGBoost** (highest PR-AUC). It caught 76 of 95 frauds with 27 false alarms.

- Random Forest was almost as good and the most precise (only 2 false alarms) but missed more fraud.
- Logistic Regression with SMOTE or class weights caught 87% of fraud but raised about 1,400 to 1,500 false alarms, so precision was very low.
- The Decision Tree was the weakest.
- Limitation: the test set has only 95 fraud cases, so the gap between XGBoost and Random Forest is small.

## How to Run
Install the dependencies with: pip install -r requirements.txt

Then run the notebooks in order (01, 02, 03) in Jupyter or Google Colab.

## Authors
- Areebah Mahmood - [@areebah-mahmood](https://github.com/areebah-mahmood)
- Sania Taqui - [SaniaTaqui28]()
