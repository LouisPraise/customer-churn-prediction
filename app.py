import streamlit as st
import joblib
import pandas as pd

# Load my files
preprocessor = joblib.load("preprocessor.pkl")
model = joblib.load("Churn_Prediction_Project.pkl")

st.title("📊 Customer Churn Prediction App")
st.write("Please enter the customer information below 👇")

# Categorical inputs

gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Has Partner", ["Yes", "No"])
Dependents = st.selectbox("Has Dependents", ["Yes", "No"])

PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)


# Numerical inputs

tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=50.0)
TotalCharges = st.text_input("Total Charges", "500")


# Prediction

if st.button("🔮 Predict Churn"):

    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [SeniorCitizen],
        "Partner": [Partner],
        "Dependents": [Dependents],
        "tenure": [tenure],
        "PhoneService": [PhoneService],
        "MultipleLines": [MultipleLines],
        "InternetService": [InternetService],
        "OnlineSecurity": [OnlineSecurity],
        "OnlineBackup": [OnlineBackup],
        "DeviceProtection": [DeviceProtection],
        "TechSupport": [TechSupport],
        "StreamingTV": [StreamingTV],
        "StreamingMovies": [StreamingMovies],
        "Contract": [Contract],
        "PaperlessBilling": [PaperlessBilling],
        "PaymentMethod": [PaymentMethod],
        "MonthlyCharges": [MonthlyCharges],
        "TotalCharges": [TotalCharges]
    })

    # Data preprocessing
    input_processed = preprocessor.transform(input_data)

    # Prediction
    prediction = model.predict(input_processed)
    probability = model.predict_proba(input_processed)[0][1]

    if prediction[0] == 1:
        st.error(f"⚠️ High risk of churn (probability = {probability:.2f})")
    else:
        st.success(f"✅ Customer likely to stay (churn probability = {probability:.2f})")
