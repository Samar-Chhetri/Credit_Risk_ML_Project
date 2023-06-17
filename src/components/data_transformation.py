import sys, os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object


class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            categorical_columns = ['Sex', 'Housing', 'Saving_accounts', 'Purpose']
            checking_account_column = ['Checking_account']
            numerical_columns = ['Age', 'Job', 'Credit_amount', 'Duration']


            cat_pipeline_1 = Pipeline([
                ('si', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder()),
                ('ss', StandardScaler(with_mean=False))

            ])

            cat_pipeline_2 = Pipeline([
                ('si', SimpleImputer(strategy='constant', fill_value=(np.random.choice(pd.Series(['little','moderate','rich']))))),
                ('ohe', OneHotEncoder()),
                ('ss', StandardScaler(with_mean=False))
            ])

            num_pipeline = Pipeline([
                ('si', SimpleImputer(strategy='median')),
                ('ss', StandardScaler(with_mean=False))
            ])

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Checking account columns: {checking_account_column}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer([
                ('cat_pipeline_1', cat_pipeline_1, categorical_columns),
                ('cat_pipeline_2', cat_pipeline_2, checking_account_column),
                ('num_pipeline', num_pipeline, numerical_columns)
            ])

            return preprocessor


        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self,train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            count = train_df['Purpose'].value_counts()
            threshold =50
            rep = count[count <threshold].index
            train_df['Purpose'] = train_df['Purpose'].replace(rep, 'others')

            count = test_df['Purpose'].value_counts()
            threshold =50
            rep = count[count <threshold].index
            test_df['Purpose'] = test_df['Purpose'].replace(rep, 'others')




            preprocessing_obj = self.get_data_transformer_object()
            logging.info("Applying preprocessing object to train and test data")

            target_column = ['Risk']

            input_feature_train_df = train_df.drop(columns=target_column).drop(columns=['Unnamed: 0'])
            target_feature_train_df = train_df['Risk']

            input_feature_test_df = test_df.drop(columns=target_column).drop(columns=['Unnamed: 0'])
            target_feature_test_df = test_df['Risk']

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            le = LabelEncoder()
            target_feature_train_arr = le.fit_transform(target_feature_train_df)
            target_feature_test_arr = le.transform(target_feature_test_df)

            input_feature_train_transformed_df = pd.DataFrame(input_feature_train_arr)
            input_feature_train_transformed_df['target'] = target_feature_train_arr
            train_arr = np.array(input_feature_train_transformed_df)

            input_feature_test_transformed_df = pd.DataFrame(input_feature_test_arr)
            input_feature_test_transformed_df['target'] = target_feature_test_arr
            test_arr = np.array(input_feature_test_transformed_df)

            logging.info("Saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            



        except Exception as e:
            raise CustomException(e, sys)
