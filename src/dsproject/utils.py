import sys
import os
from src.dsproject.exception import CustomException
from src.dsproject.logger import logging
import pandas as pd
from dotenv import load_dotenv
import pymysql
import pickle
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


load_dotenv()

host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db")

def read_sql_data():
    logging.info("Reading SQL database started")
    try:
        mydb= pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info("Connected to SQL database ",mydb)
        df=pd.read_sql_query('SELECT * FROM students',mydb)
        print(df.head())

        return df
    except Exception as e:
        raise CustomException(e,sys)
    

def save_obj(obj,filepath):
    try:
        dir_path = os.path.dirname(filepath)
       
        os.makedirs(dir_path,exist_ok=True)

        with open(filepath,'wb') as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models,params):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]
            
            grid_search = GridSearchCV(estimator=model, param_grid=param, cv=3)
            grid_search.fit(X_train,y_train) # Train Model


            model.set_params(**grid_search.best_params_)
            model.fit(X_train,y_train) # Train Model

            y_train_pred = model.predict(X_train) # Predict Train Data
            y_test_pred = model.predict(X_test) # Predict Test Data

            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]]  = test_model_score

        return report
    except Exception as e:
        raise CustomException(e,sys)
    


def load_object(file_path):
    try:
        logging.info("Loading object from path ",file_path)
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
        
        
    except Exception as e:
        raise CustomException(e,sys)