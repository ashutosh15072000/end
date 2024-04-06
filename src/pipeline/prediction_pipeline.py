import os,sys
import pandas as pd
from src.exception.exception import customexception
from src.logger.logging import logging
from src.utils.utils import load_object

class PredictPipeline:
    def __init__(self):
        print("init... the object")

    def predict(self,features):
        try:
            ## COLLECTING A PATH FOR PREPROCESSOR AND MODEL

            preprocessor_path=os.path.join("artifact",'preprocessor.pkl')
            model_path=os.path.join('artifact','model.pkl')

            ## lOAD THE PREPROCESSOR AND MODEL

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)
            ## SCALING
            scaled_feature=preprocessor.transform(features)

            pred=model.predict(scaled_feature)

            return pred


        except Exception as e:
            raise customexception (e,sys)
        
class CustomData:
    def __init__(self,
                 carat:float,
                 depth:float,
                 table:float,
                 x:float,
                 y:float,
                 z:float,
                 cut:str,
                 color:str,
                 clarity:str):
        self.carat=carat
        self.depth=depth
        self.table=table
        self.x=x
        self.y=y
        self.z=z
        self.cut=cut
        self.color=color
        self.clarity=clarity


    def get_data_as_dataframe(self):
        
        ## MAKING THE DICT FROM TAKING THE DATA FROM PREDCITON HTML PAGE

        try:
            custom_data_input_dict={
                'carat':[self.carat],
                'depth':[self.depth],
                'table':[self.table],
                'x':[self.x],
                'y':[self.y],
                'z':[self.z],
                'cut':[self.cut],
                'color':[self.color],
                'clarity':[self.clarity]
            }
            ## CONVERTING THE DICTIONARY INTO DATAFRAME

            df=pd.DataFrame(custom_data_input_dict)
            logging.info("DataFrame Gathered")
            return df
  
        except Exception as e:
            logging.info("Exception Occured in Prediction Pipeline")
            raise customexception(e,sys) 