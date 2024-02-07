import os
import sys
import pickle
import dill
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, f1_score
from src.ml_banking.exception import CustomException
from src.ml_banking.logger import logging
from dotenv import load_dotenv
import pymysql

# Load environment variables
load_dotenv()
host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv('db')

def read_sql_data():
    logging.info('Reading Sql Database started')
    try:
        mydb = pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info("Connection Established", mydb)
        df = pd.read_sql_query('Select * from transactions', mydb)
        print(df.head())
        return df
    except Exception as ex:
        raise CustomException(ex)
    
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            #model_name = list(models.keys())[i]
            model = list(models.values())[i]
            #params = param[model_name]
            params = param[list(models.keys())[i]]
            
            gs = GridSearchCV(model, params, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            if len(set(y_train)) == 2:
                # Binary classification
                train_model_score = f1_score(y_train, y_train_pred)
                test_model_score = f1_score(y_test, y_test_pred)
            else:
                # Multiclass classification
                train_model_score = classification_report(y_train, y_train_pred)
                test_model_score = classification_report(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)    