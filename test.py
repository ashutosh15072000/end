from src.components.data_transformation import DataTransformation

data_obj=DataTransformation()

data_obj.initiate_data_transformation('artifact\train.csv','artifact\test.csv')


print(data_obj)