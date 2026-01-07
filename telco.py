import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background-color: #f5f7fb;
}
.card {
    background: linear-gradient(135deg, #e0f2fe, #fef3c7);
    padding: 22px;
    border-radius: 16px;
    margin-bottom: 22px;
    box-shadow: 0 8px 18px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown(
    "<div class='card'><h2>Telco Customer Churn Prediction</h2>"
    "<p>Logistic Regression Model</p></div>",
    unsafe_allow_html=True
)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    path = os.path.join(os.path.dirname(__file__),
                        "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = pd.read_csv(path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    return df

df = load_data()

# ---------------- DATA PREVIEW ----------------
with st.expander(" Dataset Preview"):
    st.dataframe(df.head())

# ---------------- FEATURES & TARGET ----------------
y = df['Churn']
X = df.drop(columns=['Churn', 'customerID'], errors='ignore')
X = pd.get_dummies(X, drop_first=True)

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------- MODEL ----------------
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)
model.fit(X_train, y_train)

# ---------------- PREDICTIONS ----------------
y_pred = model.predict(X_test)

# ---------------- METRICS ----------------
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label="Yes")
recall    = recall_score(y_test, y_pred, pos_label="Yes")
f1        = f1_score(y_test, y_pred, pos_label="Yes")

# ---------------- METRICS DISPLAY ----------------
st.markdown("<div class='card'><h3> Model Performance</h3>", unsafe_allow_html=True)

st.metric("Accuracy", f"{accuracy:.3f}")
st.metric("Precision", f"{precision:.3f}")
st.metric("Recall", f"{recall:.3f}")
st.metric("F1 Score", f"{f1:.3f}")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- CONFUSION MATRIX HEATMAP ----------------
st.markdown("<div class='card'><h3> Confusion Matrix Heatmap</h3></div>",
            unsafe_allow_html=True)

cm = confusion_matrix(y_test, y_pred, labels=["Yes", "No"])

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Predicted Yes", "Predicted No"],
    yticklabels=["Actual Yes", "Actual No"],
    ax=ax
)
ax.set_xlabel("Prediction")
ax.set_ylabel("Actual")

st.pyplot(fig)

# ---------------- CLASSIFICATION REPORT ----------------

report_dict = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

report_df = pd.DataFrame(report_dict).transpose()

st.markdown("<div class='card'><h3>Classification Report</h3></div>",
            unsafe_allow_html=True)

st.dataframe(
    report_df.style.format("{:.3f}")
)
# ================= CUSTOMER INPUT PREDICTION =================
st.markdown(
    "<div class='card'><h3>Predict Churn for a Customer</h3>"
    "<p>Modify values to see churn risk</p></div>",
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.slider("Tenure (Months)", 0, 72, 12)

with col2:
    monthly_charges = st.slider(
        "Monthly Charges",
        0.0,
        float(df["MonthlyCharges"].max()),
        50.0,
        step=1.0
    )

with col3:
    total_charges = st.slider(
        "Total Charges",
        0.0,
        float(df["TotalCharges"].max()),
        500.0,
        step=10.0
    )

# ---------------- PREPARE INPUT ----------------
input_data = pd.DataFrame(0, index=[0], columns=X.columns)
input_data.at[0, "tenure"] = tenure
input_data.at[0, "MonthlyCharges"] = monthly_charges
input_data.at[0, "TotalCharges"] = total_charges

# ---------------- PREDICTION ----------------
prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1]

# ---------------- RESULT DISPLAY ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

if prediction == "Yes":
    st.error("⚠️ Customer is likely to CHURN")
else:
    st.success("✅ Customer is likely to STAY")

st.metric("Churn Probability", f"{probability:.2%}")

st.markdown("</div>", unsafe_allow_html=True)
