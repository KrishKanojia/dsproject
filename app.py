from src.dsproject.logger import logging
from src.dsproject.exception import CustomException
from src.dsproject.components.data_ingestion import DataIngestion,DataIngestionConfig
from src.dsproject.components.data_transformation import DataTransformation,DataTransformationConfig

import sys

if __name__ == "__main__":
    logging.info("Hello World!")

    try:
        data_ingestion = DataIngestion()
        train_data_path , test_data_path =  data_ingestion.initiate_data_ingestion()

        data_transformation = DataTransformation()
        data_transformation.initaite_data_transformation(train_path=train_data_path,test_path=test_data_path)

        
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)