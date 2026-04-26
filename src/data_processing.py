import os
import sys
import pandas as pd
import pickle

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


class DataProcessing():
    def __init__(self):
        os.makedirs(os.path.dirname('model/'),exist_ok=True)
        self.preprocessing = 'models/preprocessor.pkl'

    def get_preprocesor(self,df):
        print("Creating Prepocessing pipeline...")

        # identify columns
        numerical_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        # remove target
        if 'Churn' in numerical_cols:
            numerical_cols.remove('Churn')
        if 'Churn' in categorical_cols:
            categorical_cols.remove('Churn')

        print("Numerical Columns:", numerical_cols)
        print('Categorical Columns:', categorical_cols)

        # numerical pipeline

        num_pipeline = Pipeline(steps=[
            ('imputer',SimpleImputer(strategy='median')),
            ('scaler',StandardScaler())
        ])

        # categorical_pipeline

        cat_pipeline = Pipeline(steps=[
            ('imputer',SimpleImputer(strategy='most_frequent')),
            ('encoder',OneHotEncoder(handle_unknown='ignore'))
        ])

        # combile pipeline
        preprocessor = ColumnTransformer([
            ('num',num_pipeline,numerical_cols),
            ('cat',cat_pipeline,categorical_cols)
        ])

        return preprocessor
    
    def initiate_data_processing(self, train_path, test_path):
        print("start Data Processing....")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # dropping customer id : because it has unique value no impact on model training
        train_df = train_df.drop(columns=['customerID'],errors='ignore')
        test_df = test_df.drop(columns=['customerID'],errors='ignore')

        train_df["Churn"] = train_df["Churn"].map({"No": 0, "Yes": 1})
        test_df["Churn"] = test_df["Churn"].map({"No": 0, "Yes": 1})

        target_column = 'Churn'
        
        # train test
        X_train = train_df.drop(columns=[target_column])
        y_train = train_df[target_column]

        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]

        # get preprocessing object
        preprocessor = self.get_preprocesor(train_df)

        # fit and transform
        X_train_transform = preprocessor.fit_transform(X_train)
        X_test_transform = preprocessor.transform(X_test)

        # Save preprocessor
        os.makedirs('models', exist_ok=True)

        with open(self.preprocessing, 'wb') as f:
            pickle.dump(preprocessor, f)

        print("Preprocessor saved")

        print("Preprocessing Done")

        return (
            X_train_transform,
            X_test_transform,
            y_train,
            y_test,
            self.preprocessing
        )

if __name__ == "__main__":
    from data_ingestion import DataIngestion

    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    processor = DataProcessing()
    processor.initiate_data_processing(train_path,test_path)