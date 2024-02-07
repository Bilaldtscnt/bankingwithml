# prediction_pipeline.py
import sys
import os
import pandas as pd
from src.ml_banking.exception import CustomException
from src.ml_banking.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass


    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds    
    
        except Exception as e:
            raise CustomException(e,sys)
            
        


           

class CustomData:
    def __init__(self, TRANSACTION_ID: int, TX_DATETIME: object, CUSTOMER_ID: int, TERMINAL_ID: int, TX_AMOUNT: float, TX_TIME_SECONDS: int, TX_TIME_DAYS: int, TX_FRAUD_SCENARIO: int):
        self.TRANSACTION_ID = TRANSACTION_ID
        self.TX_DATETIME = TX_DATETIME
        self.CUSTOMER_ID = CUSTOMER_ID
        self.TERMINAL_ID = TERMINAL_ID
        self.TX_AMOUNT = TX_AMOUNT 
        self.TX_TIME_SECONDS = TX_TIME_SECONDS
        self.TX_TIME_DAYS = TX_TIME_DAYS
        self.TX_FRAUD_SCENARIO = TX_FRAUD_SCENARIO

        
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {    
                "TRANSACTION_ID":[self.TRANSACTION_ID],
                "TX_DATETIME":[self.TX_DATETIME],
                "CUSTOMER_ID":[self.CUSTOMER_ID],
                "TERMINAL_ID":[self.TERMINAL_ID],
                "TX_AMOUNT":[self.TX_AMOUNT],
                "TX_TIME_SECONDS":[self.TX_TIME_SECONDS],
                "TX_TIME_DAYS":[self.TX_TIME_DAYS],
                "TX_FRAUD_SCENARIO":[self.TX_FRAUD_SCENARIO],
            }
            
            return pd.DataFrame(custom_data_input_dict)    
        except Exception as e:
            raise CustomException(e, sys)


    