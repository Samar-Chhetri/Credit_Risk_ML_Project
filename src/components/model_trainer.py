import sys, os
import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import f1_score

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split train and test data")

            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Support Vector Classifier": SVC(),
                "K-nearest Classifier": KNeighborsClassifier()
            }

            params = {
                "Decision Tree": {
                    'criterion': ['gini', 'entropy', 'log_loss']
                },

                "Logistic Regression": {},

                "Random Forest": {
                    'criterion': ['gini', 'entropy', 'log_loss'],
                    'max_depth': [3,4,5,6,7]
                },

                "AdaBoost": {
                    'learning_rate': [0.1, 0.05, 0.5, 0.01]
                },

                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.05],
                    'criterion': ['friedman_mse', 'squared_error']
                },

                "Support Vector Classifier": {
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
                },

                "K-nearest Classifier":{
                    'n_neighbors': [3,4,5,6,7]
                }
            }



            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train,X_test=X_test,y_test=y_test, models=models, param=params)

            # Best model score from dict
            best_model_score = max(sorted(model_report.values()))
            
            # Best model name from dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score <0.6:
                raise CustomException("Best model not found")
            
            logging.info("Best model found on both training ans testing dataset")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)

            f1_sco = f1_score(y_test, predicted)
            return f1_sco
        

        except Exception as e:
            raise CustomException(e, sys)