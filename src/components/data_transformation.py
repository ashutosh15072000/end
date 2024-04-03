import pandas as pd
import numpy as np

from src.logger.logging import logging
from src.exception.exception import customexception
import os
import sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path
from src.utils.utils import save_object
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

@dataclass
## config class is used for saving or if we are 
## getting any type output that we are keep somewhere
class DataTransformationconfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")
    pass
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationconfig()
    
    def get_data_transformation(self):
        try:
            logging.info("Data Transformation initiated")

            ## DEFINE WHICH COLUMNS SHOULD BE ORDINAL-ENCODED AND WHICH SHOUB BE SCALED
            categorical_cols=['cut','color','clarity']
            numerical_cols=['carat','depth','table','x','y','z']

            ## DEFINE THE CUSTOM RANKING FOR EACH ORDINAL VARIABLE
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info("Pipeline Initiated")

            ## NUMERICAL PIPELINE 

            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            ## CATEGORICAL PIPELINE

            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                    ('scaler',StandardScaler())

                ]
            )

            ## COLUMN TRANSFORMER

            preprocessor=ColumnTransformer(
                ('num_pipeline',num_pipeline,numerical_cols),
                ('cat_pipeline',cat_pipeline,categorical_cols)
            )

            return preprocessor
        
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")
            return customexception(e,sys)
        
    def initiate_data_ingestion(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read Train and Test Data Complete")
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')

            ## CREATING THE OBJECT OF GET_DATA_TRANSFORMATION

            preprocessing_obj = self.get_data_transformation()

            target_column_name='price'
            drop_columns=[target_column_name,'id']
            ## SEGRATTING  THE TRAIN AND TARGET

            input_feature_train_df=train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            ## SEGRATTING  THE TEST AND TARGET

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            ## APPYING PREPROCESSING STEP AND CALLING FUNCTION GET_DATA_TRANSFORMATION

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            logging.info("Applying preprocessing object on training and testing datasets.")
            
            ## CONCATING THE TRAIN,TEST AND TARGET COLUMN

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            ## SAVING THE PREPROCESSING OBJ
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            logging.info("Preprocessing Pickle File Saved")
            
            return (
                train_arr,
                test_arr
            )

        except Exception as e:
            logging.info()
            raise customexception(e,sys)
