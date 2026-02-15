📌 Customer Churn Prediction API
🔹 Overview

End-to-end machine learning project that predicts customer churn using Logistic Regression with proper preprocessing pipeline and FastAPI deployment.

🔹 Tech Stack

Python

Pandas

Scikit-learn

FastAPI

Uvicorn

🔹 Model

Logistic Regression

Class imbalance handled using class_weight="balanced"

OneHotEncoding via ColumnTransformer

Full preprocessing + model saved as Pipeline

🔹 Dataset

Telco Customer Churn dataset (7,043 records, 21 features)

🔹 API Endpoint

POST /predict

Example Request:
json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 5,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 89.1,
  "TotalCharges": 445.5
}


Response:
json

{
  "prediction": 1,
  "result": "Churn",
  "churn_probability": 0.742
}

🔹 How To Run
pip install -r requirements.txt
uvicorn app.main:app --reload


