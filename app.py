
# Import required libraries

import streamlit as st              
import joblib                        
import pandas as pd                  
import numpy as np                   
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# Load the trained model

model = joblib.load("rf_model_final.pkl")



# List of categorical columns to be encoded
Yes_No_Col = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
    'InternetService', 'Contract', 'PaymentMethod', 'SeniorCitizen'
]

# Define the categorical encoder
encoder = OneHotEncoder(
    drop='first',
    handle_unknown='ignore',
    sparse_output=False
)

# Global preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', encoder, Yes_No_Col),  
        ('num', MinMaxScaler(), ['tenure', 'MonthlyCharges', 'TotalCharges'])  
    ],
    remainder='passthrough'           
)

# Function to train and cache the preprocessor
@st.cache_resource
def get_preprocessor():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    X = df.drop(['customerID', 'Churn'], axis=1)
    X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce').fillna(0)
    preprocessor.fit(X)
    return preprocessor
trained_preprocessor = get_preprocessor()



# Streamlit User Interface
st.title("📊 Customer Churn Prediction App")
st.write("Please enter the customer information below 👇")

# Categorical user inputs
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

# Numerical user inputs
tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=50.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=500.0)



# Prediction 
if st.button("🔮 Predict Churn"):
    data_dict = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }

    # Convert input data into a DataFrame
    input_df = pd.DataFrame([data_dict])

    # Apply preprocessing to user input
    input_processed_df = pd.DataFrame(
        trained_preprocessor.transform(input_df),
        columns=trained_preprocessor.get_feature_names_out()
    )

    # Select the exact features expected by the model
    cols_23 = [
        'cat__gender_Male', 'cat__Partner_Yes', 'cat__Dependents_Yes',
        'cat__PhoneService_Yes', 'cat__MultipleLines_Yes',
        'cat__OnlineSecurity_Yes', 'cat__OnlineBackup_Yes',
        'cat__DeviceProtection_Yes', 'cat__TechSupport_Yes',
        'cat__StreamingTV_Yes', 'cat__StreamingMovies_Yes',
        'cat__PaperlessBilling_Yes', 'cat__InternetService_Fiber optic',
        'cat__InternetService_No', 'cat__Contract_One year',
        'cat__Contract_Two year',
        'cat__PaymentMethod_Credit card (automatic)',
        'cat__PaymentMethod_Electronic check',
        'cat__PaymentMethod_Mailed check',
        'cat__SeniorCitizen_1',
        'num__tenure', 'num__MonthlyCharges', 'num__TotalCharges'
    ]

    final_input = input_processed_df[cols_23]

    # Make prediction and get churn probability
    prediction = model.predict(final_input)
    probability = model.predict_proba(final_input)[0][1]

    # Display result to the user
    if prediction[0] == 1:
        st.error(f"⚠️ High risk of churn ({probability:.2%})")
    else:
        st.success(f"✅ Loyal Customer ({probability:.2%})")