
import time
import sys
import os
from dotenv import load_dotenv
from flask import Flask, render_template, request
from src.ml_banking.components.data_ingestion import DataIngestion
from src.ml_banking.components.data_transformation import DataTransformation
from src.ml_banking.components.model_tranier import ModelTrainer
from src.ml_banking.exception import CustomException
from src.ml_banking.logger import logging
from src.ml_banking.pipelines.prediction_pipeline import CustomData, PredictPipeline
from pydantic import BaseModel

class MyModel(BaseModel):
    my_model_alias: str = 'default_value'

app = Flask(__name__)


# ... (Your existing routes and functions)

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint for handling autocomplete requests and form submissions
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            data = CustomData(
                TRANSACTION_ID=request.form.get('TRANSACTION_ID'),
                TX_DATETIME=request.form.get('TX_DATETIME'),
                CUSTOMER_ID=request.form.get('CUSTOMER_ID'),
                TERMINAL_ID=request.form.get('TERMINAL_ID'),
                TX_AMOUNT=request.form.get('TX_AMOUNT'),
                TX_TIME_SECONDS=request.form.get('TX_TIME_SECONDS'),
                TX_TIME_DAYS=request.form.get('TX_TIME_DAYS'),
                TX_FRAUD_SCENARIO=request.form.get('TX_FRAUD_SCENARIO')
            )

            new_data = data.get_data_as_data_frame()

            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(new_data)

            return render_template('home.html', results=results[0])

        except Exception as e:
            logging.exception("Error in prediction")
            return render_template('error.html', error_message=str(e))


# Add the block to measure execution time and log details
if __name__ == "__main__":
    # The following block is added for GCP deployment
    import os
    port = int(os.environ.get('PORT', 8080))
    host = '127.0.0.1'

    start_time = time.time()
    logging.info("The execution has started")

    try:
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        data_transformation = DataTransformation()
        train_arr, test_arr = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr, test_arr))

    except Exception as e:
        logging.exception("Custom Exception")
        raise CustomException(e, sys)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution Time: {execution_time} seconds")

    # The following line is updated for GCP deployment
    app.run(host=host, port=port, debug=False)