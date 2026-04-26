import pickle
import pandas as pd

class PredictPipeline:

    def __nit__(self):
        self.model_path = 'model/model.pkl'
        self.tune_model_path = 'model/tune_model.pkl'
        self.preprocessor_path = 'model/preprocessor.pkl'

    def load_object(self):
        print("Loading model and preprocessor...")

        with open(self.model_path, "rb") as f:
            model = pickle.load(f)

        with open(self.tune_model_path,'rb') as f:
            tune_model = pickle.load(f)

        with open(self.preprocessor_path, "rb") as f:
            preprocessor = pickle.load(f)

        return model, tune_model,preprocessor
    
    def predict(self,input_data: dict):
        try:
            input_df = pd.DataFrame([input_data])

            #load Model 
            model, tune_model, preprocessor = self.load_object()

            # transfor input data
            input_tf = preprocessor.transform(input_df)

            # Predict
            prediction = model.predict(input_tf)
            tune_prediction = tune_model.predict(input_tf)

            return prediction[0], tune_prediction[0]
        
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None
        
if __name__ == "__main__":
    # Example input (change based on your dataset features)
    sample_input = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 845.5
    }

    pipeline = PredictPipeline()
    result, result_2 = pipeline.predict(sample_input)

    print("Prediction:", result, result_2)



