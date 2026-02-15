# Breast Cancer Classification Project

## Problem Statement
Breast cancer is one of the most common cancers among women worldwide. Early detection is crucial for successful treatment. This project aims to build a Machine Learning classification model to predict whether a breast mass is **Malignant** (cancerous) or **Benign** (non-cancerous) based on digitized image features of a Fine Needle Aspirate (FNA) of a breast mass. We will explore multiple classification algorithms to find the most accurate model.

## Dataset Description
The dataset used is the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset, available via `sklearn.datasets`.
This dataset satisfies the assignment criteria:
- **Source**: UCI Machine Learning Repository (publicly available).
- **Instances**: 569 (Requirement: > 500)
- **Features**: 30 numeric, predictive attributes (Requirement: > 12)
- **Classes**: Malignant (WDBC-Malignant), Benign (WDBC-Benign)

## Models Used
The following 6 Machine Learning models were trained and evaluated:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (KNN)
4. Naive Bayes (Gaussian)
5. Random Forest Classifier
6. XGBoost Classifier

## Comparison Table
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9957 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree Classifier | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| K-Nearest Neighbor (KNN) | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes (Gaussian) | 0.9474 | 0.9881 | 0.9342 | 0.9861 | 0.9595 | 0.8872 |
| Random Forest Classifier | 0.9474 | 0.9944 | 0.9583 | 0.9583 | 0.9583 | 0.8869 |
| XGBoost Classifier | 0.9474 | 0.9940 | 0.9459 | 0.9722 | 0.9589 | 0.8864 |

## Observations
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression |  |
| Decision Tree Classifier |  |
| K-Nearest Neighbor (KNN) |  |
| Naive Bayes (Gaussian) |  |
| Random Forest Classifier |  |
| XGBoost Classifier |  |
