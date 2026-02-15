import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import os

# Set page config
st.set_page_config(page_title="Breast Cancer Classification", layout="wide")

# Title and Description
st.title("Breast Cancer Classification App")
st.write("""
This application classifies Breast Cancer tumors as Malignant or Benign using various Machine Learning models.
Upload a CSV file to test the models, or use the default test dataset.
""")

# Load models and feature names
@st.cache_resource
def load_resources():
    model_dir = 'model'
    models = {}
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl') and f != 'feature_names.pkl']
    
    for file in model_files:
        model_name = file.replace('.pkl', '').replace('_', ' ').title()
        # Fix specific naming for consistency with training script
        if "Knn" in model_name: model_name = "K-Nearest Neighbor (KNN)"
        if "Naive Bayes" in model_name: model_name = "Naive Bayes (Gaussian)"
        if "Xgboost" in model_name: model_name = "XGBoost Classifier"
        
        models[model_name] = joblib.load(os.path.join(model_dir, file))
        
    feature_names = joblib.load(os.path.join(model_dir, 'feature_names.pkl'))
    return models, feature_names

try:
    models, feature_names = load_resources()
except FileNotFoundError:
    st.error("Models not found. Please run the training script first.")
    st.stop()

# Sidebar - Model Selection
st.sidebar.header("Model Selection")
selected_model_name = st.sidebar.selectbox("Choose a Model", list(models.keys()))
model = models[selected_model_name]

# Main Area - Data Upload
st.header("1. Data Loading")
uploaded_file = st.file_uploader("Upload a CSV file (features must match dataset)", type="csv")

X_test = None
y_test = None

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        # Check if target is present
        if 'target' in df.columns:
            y_test = df['target']
            X_test = df.drop('target', axis=1)
        else:
            # Assuming all columns are features if no target
            X_test = df
            st.warning("No 'target' column found. Metrics cannot be calculated, only predictions.")
            
        # Ensure feature columns match
        # This checks intersection of columns. 
        # Ideally, we should validate strictly against feature_names but for flexibility we'll find common ones or use all if matching.
        # For this assignment, let's assume valid input or fall back to standard validation.
        # We will reindex to ensure order matches training
        missing_cols = set(feature_names) - set(X_test.columns)
        if missing_cols:
             st.error(f"Uploaded file is missing features: {missing_cols}")
             X_test = None
        else:
             X_test = X_test[feature_names]

    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    st.info("No file uploaded. Using default test set (20% of sklearn breast_cancer dataset).")
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    # Stratify split to match training Logic
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Add a reference table for model performance in the sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Model Performance Leaderboard")
performance_data = {
    "Model": ["Logistic Regression", "KNN", "Random Forest", "Naive Bayes", "XGBoost", "Decision Tree"],
    "Accuracy": [0.9825, 0.9737, 0.9474, 0.9474, 0.9474, 0.9211],
    "F1 Score": [0.9861, 0.9796, 0.9583, 0.9595, 0.9589, 0.9362]
}
perf_df = pd.DataFrame(performance_data).sort_values(by="Accuracy", ascending=False)
st.sidebar.dataframe(perf_df.set_index("Model"), height=250)

# Prediction and Evaluation
if X_test is not None:
    st.header(f"2. Results for {selected_model_name}")
    
    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    # Display Metrics if y_test is available
    if y_test is not None:
        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("AUC Score", f"{auc:.4f}")
        col3.metric("Precision", f"{precision:.4f}")
        col4.metric("Recall", f"{recall:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC", f"{mcc:.4f}")
        
        # Visualizations
        st.header("3. Visualizations")
        
        tab1, tab2 = st.tabs(["Confusion Matrix", "Classification Report"])
        
        with tab1:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(6, 4))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig, use_container_width=False)
            
        with tab2:
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())
            
    else:
        st.subheader("Predictions")
        st.write(y_pred)

else:
    st.write("Waiting for valid data...")
