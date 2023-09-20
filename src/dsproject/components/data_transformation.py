import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.dsproject.utils import save_obj
from src.dsproject.exception import CustomException
from src.dsproject.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation(self):
        '''
        This function is reponsible for Data Transformation
        '''
        try:
            numerical_columns = ["reading_score","writing_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "test_preparation_course",
                "lunch"
            ]

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
            ])

            logging.info(f"Categorical Columns: {categorical_columns}")
            logging.info(f"Numerical Columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)


    
    def initaite_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading Train and Test file")

            precossing_obj = self.get_data_transformation()

            target_column_name = "math_score"
            numerical_columns = ["reading_score","writing_score"]

            ## Divide Dataset into Dependent and Independent Features
            input_feature_train_df = train_df.drop([target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop([target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying Preprocessing on Train and Test Data")

            transformed_feature_train_arr =  precossing_obj.fit_transform(input_feature_train_df)
            transformed_feature_test_arr = precossing_obj.transform(input_feature_test_df)

            train_arr = np.c_[transformed_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_feature_test_arr,np.array(target_feature_test_df)]

            logging.info("Saved Preprocessing Object")

            save_obj(
                filepath=self.data_transformation_config.preprocessor_obj_file_path,
                obj=precossing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
