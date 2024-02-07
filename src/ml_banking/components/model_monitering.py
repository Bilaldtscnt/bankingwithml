
'''
import mlflow
import pandas as pd
import numpy as np
from src.ml_banking.exception import CustomException
from src.ml_banking.logger import logging
from src.ml_banking.components.model_monitering import ModelMonitoring
from src.ml_banking.components.model_monitering import ModelMonitoringConfig
from src.ml_banking.utils import read_sql_data
from datetime import datetime
import time

# Define the frequency at which you want to monitor the model (e.g., daily)
monitoring_frequency = 24 * 60 * 60  # 24 hours

# Define the model performance threshold for alerting (e.g., if F1-score drops below 0.8)
performance_threshold = 0.8

if __name__ == "__main__":
    logging.info("Model Monitoring script started")

    while True:
        try:
            # Initialize MLflow and Dagshub
            mlflow.set_registry_uri("https://dagshub.com/KhanBilal29/mlbanking.mlflow")
            mlflow.start_run()

            # Get the latest version of the registered model
            model_version = mlflow.search_runs(filter_string='tags.mlflow.source.type = "REGISTERED"').iloc[0]['artifact_uri'].split("/")[-1]

            # Load the latest registered model
            model = mlflow.sklearn.load_model(model_uri=f"models:/{model_version}")

            # Fetch new data for monitoring (assuming you have a data source)
            new_data = read_sql_data()

            # Perform data transformation on new data
            model_monitoring = ModelMonitoring()
            transformed_data = model_monitoring.transform_data(new_data)

            # Make predictions using the model
            predicted_qualities = model.predict(transformed_data)

            # Evaluate model performance on new data
            f1_score, _, _, _ = model_monitoring.eval_metrics(new_data["TX_FRAUD"], predicted_qualities)

            logging.info(f"Model version: {model_version}")
            logging.info(f"F1 Score on new data: {f1_score}")

            # Log the model performance metrics
            mlflow.log_metric("F1 Score", f1_score)
            mlflow.log_param("Model Version", model_version)
            mlflow.log_param("Monitoring Frequency (seconds)", monitoring_frequency)

            # Alert if the model's performance is below the threshold
            if f1_score < performance_threshold:
                logging.warning("Model performance has dropped below the threshold. Alerting!")
                # Add your alerting mechanism here (e.g., sending an email, Slack message, etc.)

            mlflow.end_run()

            logging.info("Model Monitoring completed")

        except CustomException as ce:
            logging.error(f"Custom Exception: {ce}")
        except Exception as e:
            logging.error(f"An error occurred: {e}")

        # Sleep for the monitoring frequency before the next check
        time.sleep(monitoring_frequency)

'''