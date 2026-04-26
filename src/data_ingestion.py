import os
import pandas as pd
from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self):
        self.raw_data_path = 'data/raw.csv'
        self.train_data_path = 'data/train.csv'
        self.test_data_path = 'data/test.csv'

    def initiate_data_ingestion(self):
        print("starting data ingestion....")

        # Load data 
        df = pd.read_csv(self.raw_data_path)
        print("Dataset loaded")

        # train_test split
        train_set, test_set = train_test_split(df,test_size=0.2,random_state=42)

        # craete dictonary if not created
        os.makedirs(os.path.dirname(self.train_data_path),exist_ok=True)

        # save files 
        train_set.to_csv(self.train_data_path,index=False)
        test_set.to_csv(self.test_data_path, index=False)

        print("Data ingestion Completed")
        return self.train_data_path,self.test_data_path
    

if __name__== "__main__":
    ingestion = DataIngestion()
    train_path,test_path = ingestion.initiate_data_ingestion()
    print(train_path,test_path)



