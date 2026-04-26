import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Initialize app
app = FastAPI()

# Load model & preprocessor
with open("models/tune_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)


# Input schema (VERY IMPORTANT)
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# Health check
@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}


# Prediction endpoint
@app.post("/predict")
def predict(data: CustomerData):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([data.dict()])

        # Transform
        transformed_data = preprocessor.transform(input_df)

        # Predict
        prediction = model.predict(transformed_data)[0]

        return {
            "prediction": int(prediction),
            "churn": "Yes" if prediction == 1 else "No"
        }

    except Exception as e:
        return {"error": str(e)}