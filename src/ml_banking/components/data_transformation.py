import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.ml_banking.exception import CustomException
from src.ml_banking.logger import logging
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from src.ml_banking.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            # Create a preprocessor
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer,  ('TX_AMOUNT','TX_TIME_SECONDS', 'TX_TIME_DAYS'))
                ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            preprocessor = self.get_data_transformer_object()

            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            # Define features and target
            target_column = "TX_FRAUD"
            numeric_features = train_df.select_dtypes(include=['float64', 'int64']).columns
            input_feature_train_df = train_df.drop(columns=[target_column])
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column])
            target_feature_test_df = test_df[target_column]

            logging.info("Applying Preprocessing on training and test dataframes")

            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )

            return (
                train_arr,
                test_arr,
                #self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)




