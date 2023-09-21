import sys
import os
import pandas as pd
from src.dsproject.exception import CustomException
from src.dsproject.logger import logging
from src.dsproject.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            logging.info("Prediction started")
            model_path = os.path.join("artifacts","model.pkl")
            preprocessor_path = os.path.join("artifacts","preprocessor.pkl")

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)
            prediction = model.predict(data_scaled)

            return prediction


        except Exception as e:
            raise CustomException(e,sys)





class CustomData:
    def __init__(self,
        gender:str,
        lunch:str,
        race_ethnicity:str,
        test_prep_course:str,
        parental_education,
        reading_score:int,
        writing_score:int):

        self.gender = gender

        self.lunch = lunch
        
        self.race_ethnicity = race_ethnicity
        
        self.test_prep_course = test_prep_course
        
        self.parental_education = parental_education
        
        self.reading_score = reading_score
        
        self.writing_score = writing_score



    def get_data_as_dataframe(self):
        try:
            # logging.info("Creating dataframe from custom data")
            customData_input_dict = {
                "gender" : [self.gender],
                "lunch" : [self.lunch],
                "race_ethnicity" : [self.race_ethnicity],
                "test_preparation_course" : [self.test_prep_course],
                "parental_level_of_education" : [self.parental_education],
                "reading_score" : [self.reading_score],
                "writing_score" : [self.writing_score]
            }

            return pd.DataFrame(customData_input_dict)
        
        except Exception as e:
            raise CustomException(e,sys)






