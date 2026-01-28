
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# Load my files
model = joblib.load("rf_model_final.pkl")

Yes_No_Col = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
              'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
              'StreamingTV', 'StreamingMovies', 'PaperlessBilling','InternetService',
              'Contract','PaymentMethod']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), Yes_No_Col),
        ('num', MinMaxScaler(), ['tenure', 'MonthlyCharges', 'TotalCharges'])
    ], 
    remainder='passthrough' 
)

# 3. On "entraîne" le preprocessor avec ton fichier CSV
@st.cache_resource
def get_preprocessor():
    # Charge le dataset (assure-toi que le CSV est dans ton dossier GitHub)
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    X = df.drop(['customerID', 'Churn'], axis=1)
    X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce').fillna(0)
    preprocessor.fit(X)
    return preprocessor

trained_preprocessor = get_preprocessor()

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
TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=500.0)

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
 
    # Transformation
    input_processed = trained_preprocessor.transform(input_data)

    # Prediction ET Probabilité
    prediction = model.predict(input_processed)
    probability = model.predict_proba(input_processed)[0][1] 

    if prediction[0] == 1:
        st.error(f"⚠️ High risk of churn (Probability: {probability:.2%})")
    else:
        st.success(f"✅ Likely to stay (Churn probability: {probability:.2%})")

    
   
    



