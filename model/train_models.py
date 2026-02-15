import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
import joblib
import os

# Create model directory if it doesn't exist
if not os.path.exists('model'):
    os.makedirs('model')

# Load the dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Split into Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "Decision Tree Classifier": DecisionTreeClassifier(),
    "K-Nearest Neighbor (KNN)": KNeighborsClassifier(),
    "Naive Bayes (Gaussian)": GaussianNB(),
    "Random Forest Classifier": RandomForestClassifier(),
    "XGBoost Classifier": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

print(f"{'Model':<30} | {'Accuracy':<10} | {'AUC':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10} | {'MCC':<10}")
print("-" * 100)

results = {}

for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(model, f'model/{name.replace(" ", "_").lower()}.pkl')
    
    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred # robust check
    
    # Calculate Metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    results[name] = {
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc
    }
    
    print(f"{name:<30} | {accuracy:<10.4f} | {auc:<10.4f} | {precision:<10.4f} | {recall:<10.4f} | {f1:<10.4f} | {mcc:<10.4f}")

# Save the feature names for later use in the app (crucial for ensuring input consistency)
joblib.dump(data.feature_names, 'model/feature_names.pkl')
print("\nAll models and feature names saved in 'model/' directory.")
