import os
import sys

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor
)
from sklearn.svm import SVR
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
from xgboost import XGBRegressor


from src.dsproject.exception import CustomException
from src.dsproject.logger import logging
from src.dsproject.utils import save_obj, evaluate_model



from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Split train and test transform data")

            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1], ## Remove train target column
                train_arr[:,-1], ## Target train column
                test_arr[:,:-1], ## Remove test target column
                test_arr[:,-1] ## Target test column
            )
            
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "KNeigbors": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "SVR": SVR(),
                "CatBoost": CatBoostRegressor(verbose=False),
            }
            logging.info("Models evaluation started")
            model_report: dict = evaluate_model(X_train= X_train, y_train= y_train, X_test=X_test,
                                                 y_test=y_test, models= models)
            
            ## Get best model score
            best_model_score = max(sorted(model_report.values()))

            ## Get best model name
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model")
            logging.info("Best found model on training and testing data")

            save_obj(
                filepath=self.model_trainer_config.trained_model_file_path,
                obj=best_model
                
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test,predicted)
            return r2_square
            
        except Exception as e:
            raise CustomException(e, sys)





