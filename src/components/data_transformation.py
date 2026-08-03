import os 
import sys 
from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler



## Main purpose is to data cleaning and feature enginerring
@dataclass
class data_transformation_config:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=data_transformation_config()

    def get_transformer_function(self):
        '''
        This is responsible for the data transformation technique
        '''
        try:
            numerical_feature=['reading_score', 'writing_score']
            cat_feature=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            numerical_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('standard_scaler',StandardScaler())
                ]
            )
            logging.info("Numerical data has been standerised")
            categroical_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder(sparse_output=False)),
                    ('standard_scaler',StandardScaler())
                ]
            )
            logging.info('categorical features has been encoded and is ready to use')

            ### now we have to combine both the pipelines
            preprocessor=ColumnTransformer(
                [
                    ('numerical_pipeline',numerical_pipeline,numerical_feature),
                    ('categorical_pipeline',categroical_pipeline,cat_feature)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,test_path,train_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info('Data has been read for both train and testing')

            preprocessor_obj=self.get_transformer_function()
            target_column='math_score'
            numerical_feature=['reading_score', 'writing_score']

            input_train_df=train_df.drop([target_column],axis=1)
            target_train_df=train_df[target_column]

            input_test_df=test_df.drop([target_column],axis=1)
            target_test_df=test_df[target_column]

            logging.info('applying the preprocessing data on training and testing dataframe')

            input_feature_train_done=preprocessor_obj.fit_transform(input_train_df)
            input_feature_test_done=preprocessor_obj.transform(input_test_df)

            train_arr=np.c_[
                input_feature_train_done,np.array(target_train_df)
            ]

            test_arr=np.c_[
                input_feature_test_done,np.array(target_test_df)
            ]

            save_obj(
                data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )


            return (
                train_arr,
                test_arr,
                data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e,sys)

## this is for the testing purpose

# if __name__=='__main__':
#     obj=DataIngestion()
#     test_data,train_data=obj.initiate_data_ingestion()

#     data_transform=DataTransformation()
#     data_transform.initiate_data_transformation(test_data,train_data)

        
