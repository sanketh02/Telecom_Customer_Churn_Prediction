import os
import pandas as pd
import pickle
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score,recall_score,precision_score,roc_auc_score
from sklearn.model_selection import GridSearchCV

class ModelTraining:
    def __init__(self):
        self.tune_model_path = "models/tune_model.pkl"
        self.model_path = "models/model.pkl"

    def model_building(self):
        print("Model Building.....")

        models = {
            "LogisticRegressor":LogisticRegression(max_iter=1000),
            "RandomForest":RandomForestClassifier(),
            "Xgboost":XGBClassifier(n_estimators=100)
        }

        return models
    
    def model_training(self,models,X_train,X_test,y_train,y_test):
        print("Model Training Start.......")

        best_score = 0
        best_model = None

        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("Churn Prediction")

        for name, model in models.items():

            with mlflow.start_run(run_name=name):

                model.fit(X_train,y_train)

                pred = model.predict(X_test)

                accuracy = accuracy_score(y_test,pred)
                recall = recall_score(y_test,pred)
                precision = precision_score(y_test,pred)
                roc_auc = roc_auc_score(y_test,pred)

                # log parameter
                mlflow.log_param("model_name", name)

                # log metrics
                mlflow.log_metrics({
                    "accuracy":accuracy,
                    "recall":recall,
                    "precision":precision,
                    "roc_auc":roc_auc
                })

                # Log Model
                mlflow.sklearn.log_model(model,"model")

                if accuracy > best_score:
                    best_score = accuracy
                    best_model = model
                
        return best_model
    
    def save_model(self,model):
        os.makedirs(os.path.dirname(self.model_path),exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(model, f)

        print("Best model saved successfully")

    def initiate_model_training(self,X_train,X_test,y_train,y_test):
        models = self.model_building()
        best_model = self.model_training(models,X_train,X_test,y_train,y_test)
        self.save_model(best_model)

    # Hyper-Parameter Tuning
    def model_building_with_tuning(self):
        print("Start Model Building with tuning.....")

        tune_models = {
            "LogisticRegression": {
                "model": LogisticRegression(max_iter=1000),
                "param":{
                    "C": [0.01,0.1,1,10],
                    "solver":['lbfgs']
                }
            },
            "RandomForest": {
                "model":RandomForestClassifier(),
                "param":{
                    "n_estimators" : [50,100],
                    "max_depth": [None,10,20]
                }
            },
            "xgboost":{
                "model" : XGBClassifier(),
                "param": {
                    "n_estimators":[50,100],
                    "max_depth": [None,10,20,30],
                    "learning_rate" : [0.1,0.2,0.5,0.9]
                }
            }
        }
        print("Model Building Completed...")

        return tune_models

    def tune_model_training(self,models,X_train,X_test,y_train,y_test):
        print("Start Model Training.......")

        best_model = None
        best_score = 0

        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("Churn Prediction tuning....")

        for name,model_config in models.items():
            with mlflow.start_run(run_name=name):

                grid = GridSearchCV(
                    estimator=model_config["model"],
                    param_grid=model_config["param"],
                    cv=3,
                    n_jobs=-1
                )

                grid.fit(X_train,y_train)

                best_estimator = grid.best_estimator_
                pred = best_estimator.predict(X_test)
                accuracy = accuracy_score(y_test,pred)
                

                # log best param
                mlflow.log_params(grid.best_params_)

                # log metric
                mlflow.log_metric("accuracy",accuracy)

                # log model
                if name == "xgboost":
                    mlflow.xgboost.log_model(best_estimator,"model")
                else:
                    mlflow.sklearn.log_model(best_estimator,"model")

                print(f"{name} Best Params: {grid.best_params_}")
                print(f"{name} Accuracy: {accuracy}")

                if accuracy > best_score:
                    best_score = accuracy
                    best_model = best_estimator

        return best_model
    
    def save_model_tune(self,model):

        print("Saving Model.....")

        os.makedirs(os.path.dirname(self.tune_model_path),exist_ok=True)
        with open(self.tune_model_path,"wb") as f:
            pickle.dump(model,f)

        print("Best Tune Model Save Sucessfully")

    def initiate_tune_model_training(self,X_train,X_test,y_train,y_test):
        models = self.model_building_with_tuning()
        best_model = self.tune_model_training(models,X_train,X_test,y_train,y_test)
        self.save_model_tune(best_model)

if __name__ == "__main__":
    from data_ingestion import DataIngestion
    from data_processing import DataProcessing

    # step_1
    ingestion = DataIngestion()
    train_path,test_path = ingestion.initiate_data_ingestion()
    print("Train Path:", train_path)
    print("Test Path:", test_path)

    # step_2
    processor = DataProcessing()
    X_train,X_test,y_train,y_test,processor_path = processor.initiate_data_processing(train_path,test_path)

    # step_3
    model_training = ModelTraining()
    model_training.initiate_model_training(X_train,X_test,y_train,y_test)

    # step_3
    tune_model_training = ModelTraining()
    model_training.initiate_tune_model_training(X_train,X_test,y_train,y_test)










