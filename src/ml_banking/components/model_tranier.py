import os
import sys
from dataclasses import dataclass
from urllib.parse import urlparse

import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from src.ml_banking.exception import CustomException
from src.ml_banking.logger import logging
from src.ml_banking.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def eval_metrics(self, y_test, y_pred):
        report = classification_report(y_test, y_pred, output_dict=True)
        f1_score_weighted = report["weighted avg"]["f1-score"]
        f1_score = report.get("macro avg", {}).get("f1-score", 0.0)
        precision = report["weighted avg"]["precision"]
        recall = report["weighted avg"]["recall"]
        return f1_score_weighted, f1_score, precision, recall


    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "K-Nearest Neighbors": KNeighborsClassifier(),
                "Gradient Boosting": GradientBoostingClassifier()
            }

            params = {
                "Logistic Regression": {
                    'C': [0.01, 0.1, 1],
                    'penalty': ['l2'],
                    'solver': ['lbfgs', 'liblinear'],
                    'max_iter': [100]

                 },
                 "Decision Tree": {
                     'criterion': ['gini', 'entropy'],
                 },
                 "Random Forest": {
                     'n_estimators': [32, 64, 128]
                 },
                 "K-Nearest Neighbors": {
                     'n_neighbors': [5, 7],
                     'weights': ['uniform', 'distance'],
                 },
                 "Gradient Boosting": {
                     'learning_rate': [0.1, 0.01],
                     'subsample': [0.7, 0.8],
                     'n_estimators': [64, 128]
                 }
            }

   


            model_report:dict=evaluate_models(X_train,y_train,X_test,y_test,models,params)
            best_model_score = max(sorted(model_report.values()))

            

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]


            print("This is the best model:")
            print(best_model_name)

            #model_names = list(params.keys())

            #actual_model=""

            #for model in model_names:
            #    if best_model_name == model:
            #        actual_model = actual_model + model

            #best_params = params[actual_model]        

            
            #mlflow.set_registry_uri("https://dagshub.com/KhanBilal29/mlbanking.mlflow")
            #tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme        

        

            ##mlflow
            #with mlflow.start_run():
            #    predicted_qualities = best_model.predict(X_test)

            #    (f1_score_weighted, f1_scores, precision, recall) = self.eval_metrics(y_test, predicted_qualities)

            #    mlflow.log_params(best_params)

            #    mlflow.log_metric("f1_score_weighted", f1_score_weighted)
            #    mlflow.log_metric("f1_scores", f1_scores)
            #    mlflow.log_metric("precision", precision)
            #    mlflow.log_metric("recall", recall)

            #    # Model registry does not work with file store
            #    if tracking_url_type_store != "file":

                    
            #        mlflow.sklearn.log_model(best_model, "model", registered_model_name=actual_model)
            #    else:
            #        mlflow.sklearn.log_model(best_model, "model")



            
            
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")


            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            ) 

            predicted=best_model.predict(X_test)

            #acc_score = accuracy_score(y_test, predicted)
            f_one_score = f1_score(y_test, predicted)
            return f_one_score

            #return acc_score
            # The rest of your code to save and handle the best model goes here
           
        except Exception as e:
            raise CustomException(e, sys)

           
    