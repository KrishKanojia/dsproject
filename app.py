from src.dsproject.logger import logging
from src.dsproject.exception import CustomException
from src.dsproject.components.data_ingestion import DataIngestion,DataIngestionConfig
from src.dsproject.components.data_transformation import DataTransformation,DataTransformationConfig
from src.dsproject.components.model_trainer import ModelTrainer
from flask import Flask, request, render_template
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.dsproject.pipelines.prediction_pipeline import PredictPipeline,CustomData


app = Flask(__name__)

## Route for home page
@app.route('/')
def index():
    return render_template('index.html')


## Route for prediction
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_education=request.form.get('parental_level_of_education'),
            test_prep_course=request.form.get('test_preparation_course'),
            lunch=request.form.get('lunch'),
            reading_score=request.form.get('reading_score'),
            writing_score=request.form.get('writing_score')
        )  

        data_df = data.get_data_as_dataframe()
        print(data_df)

        
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(data_df)
        print(f"Results: {results}")
        return render_template('index.html',results=round(results[0],2))


if __name__ == "__main__":
    logging.info("Hello World!")

    app.run(host='0.0.0.0')

    # try:
    #     data_ingestion = DataIngestion()
    #     train_data_path , test_data_path =  data_ingestion.initiate_data_ingestion()

    #     data_transformation = DataTransformation()
    #     train_arr, test_arr, _ = data_transformation.initaite_data_transformation(
    #         train_path=train_data_path,
    #         test_path=test_data_path
    #     )

    #     model_trainer = ModelTrainer()
    #     print(model_trainer.initiate_model_trainer(train_arr,test_arr))

    # except Exception as e:
    #     logging.info("Custom Exception")
    #     raise CustomException(e,sys)