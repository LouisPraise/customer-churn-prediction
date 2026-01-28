🚀 End-to-End Customer Churn Prediction System
📊 Project Overview
Customer attrition (churn) is one of the most critical challenges in the telecommunications industry. This project aims to predict the probability of a customer leaving the company to enable proactive retention strategies.
I developed a complete machine learning pipeline, moving from raw data exploration to a fully functional web application deployed in production.
🎯 Key Technical Achievements
Model Benchmarking: Conducted a deep comparison between Random Forest (80% Accuracy) and Neural Networks (73%). Random Forest was selected as the final model due to its superior performance on structured tabular data.
Robust Preprocessing Pipeline: Implemented a ColumnTransformer to automate data transformation, including OneHotEncoding for categorical variables and MinMaxScaler for numerical features.
Production-Ready Deployment: Developed an interactive web interface using Streamlit, allowing users to input customer data and receive instant risk assessments.
Technical Problem Solving: Successfully resolved complex MLOps challenges, including Python environment versioning (3.13 compatibility), object serialization with joblib, and feature dimension synchronization (23-feature alignment).
🛠️ Tech Stack
Languages & Libraries: Python, Pandas, NumPy, Scikit-Learn, TensorFlow.
Data Visualization: Matplotlib
Deployment & MLOps: Streamlit, Joblib, GitHub.
⚙️ Installation & Usage
Clone the repo: git clone https://github.com
Install dependencies: pip install -r requirements.txt
Run the app: streamlit run app.py
🧪 Model Performance (Random Forest)
Metric	Score
Accuracy	80%
F1-Score (Class 1)	[Ton Score, ex: 0.58]
Recall (Class 1)	[Ton Score, ex: 0.55]
🔗 Live Demo Link
