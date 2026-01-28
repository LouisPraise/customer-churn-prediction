
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
              'Contract','PaymentMethod', 'SeniorCitizen'] 

# On définit l'encodeur séparément pour plus de contrôle
encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', encoder, Yes_No_Col),
        ('num', MinMaxScaler(), ['tenure', 'MonthlyCharges', 'TotalCharges'])
    ], 
    remainder='passthrough' 
)

@st.cache_resource
def get_preprocessor():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    # Nettoyage strict
    X = df.drop(['customerID', 'Churn'], axis=1)
    X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce').fillna(0)
    
    # On entraîne le preprocessor sur TOUT le dataset original
    # Cela garantit que toutes les colonnes OneHot (30 ou 31) sont créées
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
   
    
if st.button("🔮 Predict Churn"):

    # 1. On crée le dictionnaire avec les noms exacts du .info()
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
 
    # 2. On transforme en DataFrame
    input_df = pd.DataFrame([data_dict])

    # 3. On force l'ordre EXACT du dataset d'entraînement
    column_order = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", 
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", 
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", 
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", 
        "MonthlyCharges", "TotalCharges"
    ]
    
    input_df = input_df[column_order]

    # 4. Transformation et Prédiction
    input_processed = trained_preprocessor.transform(input_df)
    
    # Debug pour vérifier que les dimensions collent enfin (ex: 1, 31)
    # st.write(f"Format après transformation : {input_processed.shape}")
    # Lignes de diagnostic temporaires
    st.write(f"📊 Le modèle attend {model.n_features_in_} colonnes.")
    st.write(f"⚙️ Le preprocessor en a généré {input_processed.shape[1]}.")

    prediction = model.predict(input_processed)
    # Bonne syntaxe pour la probabilité
    prob_array = model.predict_proba(input_processed)
    probability = prob_array[0][1]

    if prediction[0] == 1:
        st.error(f"⚠️ High risk of churn (Probability: {probability:.2%})")
    else:
        st.success(f"✅ Likely to stay (Churn probability: {probability:.2%})")






