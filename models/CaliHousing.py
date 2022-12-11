
from typing import List, Optional,Union
from pydantic import BaseModel

class model_CaliHousing():

    Predictions: list()
    Labels: list()
    rmse_train: float
    mean: float
    std: float
    rmse_test: float

    '''def __init__(self,Predictions,Labels,rmse_train,mean,std,rmse_test):
        self.Predictions = Predictions
        self.Labels = Labels
        self.rmse_train = rmse_train
        self.mean = mean
        self.std = std
        self.rmse_test = rmse_test'''

class model_predict_house(BaseModel):
    longitude : Union[float, None] = None
    latitude: Union[float, None] = None
    housing_median_age: Union[int, None] = None
    total_rooms: Union[int, None] = None
    total_bedrooms: Union[int, None] = None
    population: Union[float, None] = None
    households: Union[int, None] = None
    median_income: Union[int, None] = None
    ocean_proximity: Union[str, None] = None



