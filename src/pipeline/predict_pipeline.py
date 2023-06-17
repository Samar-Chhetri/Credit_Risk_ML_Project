import sys, os

import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PreditPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e ,sys)

class CustomData:
    def __init__(self, Sex:str, Age:int, Job:int, Housing:str, Saving_accounts:str, Checking_account:str, Credit_amount:int,
                  Duration:int, Purpose:str):
        self.Sex = Sex
        self.Age = Age
        self.Job = Job
        self.Housing = Housing
        self.Saving_accounts = Saving_accounts
        self.Checking_account = Checking_account
        self.Credit_amount = Credit_amount
        self.Duration = Duration
        self.Purpose = Purpose


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Sex": [self.Sex],
                "Age": [self.Age],
                "Job": [self.Job],
                "Housing": [self.Housing],
                "Saving_accounts": [self.Saving_accounts],
                "Checking_account": [self.Checking_account],
                "Credit_amount": [self.Credit_amount],
                "Duration": [self.Duration],
                "Purpose": [self.Purpose]
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)