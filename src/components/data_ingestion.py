import pandas as pd
import numpy as np

from src.logger.logging import logging
from src.exception.exception import customexception
import os
import sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path
@dataclass
class DataIngestionconfig:
    raw_data_path:str=os.path.join('artifact','raw.csv')
    train_data_path:str=os.path.join('artifact',"train.csv")
    test_data_path:str=os.path.join("artifact","test.csv")
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionconfig()
        
        pass
    def initiate_data_ingestion(self):
        logging.info("Data ingestion start ")
        try:
            ## READING THE DATASET

            data=pd.read_csv("https://raw.githubusercontent.com/ashutosh15072000/end/main/Experiment/train.csv")
            logging.info("Reading a DataFrame")
            
            ## MAKING THE ARTIFACT FOLDER WHERE WE SAVE OUR DATA

            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)))
            
            ## SAVING THE DATA INTO ARTIFACT FOLDER

            data.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info("I have saved the raw dataset in artifact folder")

            logging.info("Here I have performed train test split")

            ## PERFORMING TRAIN TEST SPLIT
            train_data,test_data=train_test_split(data,test_size=0.25)
            logging.info("train test split completed")
            

            ## SAVE THE TEST AND TRAIN DATA INTO ARTIFACT FOLDER
            train_data.to_csv(self.ingestion_config.train_data_path,index=False)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False)

            logging.info("Data Ingestion Part Completed")
            return (

                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info("Exception Occur at Data Ingestion")
            raise customexception(e,sys)


if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()