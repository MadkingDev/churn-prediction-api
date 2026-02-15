from fastapi import FastAPI
import pandas as pd
from app.schemas import Customer
from app.model_loader import load_model

app = FastAPI(title="Churn Prediction Service")

model = load_model()

@app.get("/")
def health():
    return {"status": "running"}

@app.post("/predict")
def predict(customer: Customer):
    data = pd.DataFrame([customer.model_dump()])
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    return {
        "prediction": int(prediction),
        "probability": round(float(probability), 3)
    }
