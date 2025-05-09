import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, roc_auc_score
)

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("🛡️ AI-Powered Credit Card Fraud Detection Dashboard")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("creditcard_cleaned.csv")

df = load_data()

# Sidebar
st.sidebar.header("🔧 Model Settings")
model_choice = st.sidebar.selectbox("Select a model", ["Logistic Regression", "Random Forest"])
show_live_prediction = st.sidebar.checkbox("Enable Live Fraud Prediction", value=True)

# Overview
st.subheader("📊 Data Overview")
st.write(df.head())
st.write("Class Distribution:", df['Class'].value_counts())

# Correlation heatmap
with st.expander("📈 Correlation Heatmap"):
    plt.figure(figsize=(10, 6))
    corr = df.corr()['Class'].sort_values(ascending=False)
    sns.barplot(x=corr.values, y=corr.index)
    st.pyplot(plt.gcf())

# Prepare features
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train model
if model_choice == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
elif model_choice == "Random Forest":
    model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluation
st.subheader("🧪 Model Performance")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
st.write("📌 Confusion Matrix:")
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
st.pyplot(fig)

# Classification report
st.write("📌 Classification Report:")
st.text(classification_report(y_test, y_pred))

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_score = roc_auc_score(y_test, y_prob)
st.write(f"📌 ROC AUC Score: {roc_score:.4f}")
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_score:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
st.pyplot(plt.gcf())

# Live prediction
if show_live_prediction:
    st.subheader("🔍 Live Fraud Prediction")

    with st.form("prediction_form"):
        v_features = [st.number_input(f"V{i}", value=0.0) for i in range(1, 29)]
        amount_scaled = st.number_input("Amount_scaled", value=0.0)
        time_scaled = st.number_input("Time_scaled", value=0.0)
        submitted = st.form_submit_button("Predict")

        if submitted:
            input_data = pd.DataFrame([v_features + [amount_scaled, time_scaled]], columns=X.columns)
            prediction = model.predict(input_data)[0]
            st.success(f"🧾 Prediction: {'Fraud' if prediction == 1 else 'Not Fraud'}")

