import json
from typing import List, Optional,Union
from pydantic import BaseModel
import models.CaliHousing as md

from topics.CaliHousing import Linear_Regression as lin_gr
from topics.CaliHousing import Decision_Tree_Regression as de_tr_gr
from topics.CaliHousing import Random_Forest_Regression as rf_gr
from topics.CaliHousing import Random_Forest_Regression_Grid_Search_CV as gs_gr
from topics.CaliHousing import Random_Forest_Regression_Random_Search_CV as rd_gr

from fastapi import APIRouter

router = APIRouter()


# Linear Regression
@router.get("/calihousing/linear-data-sample")
async def getSample():

    data =  md.model_CaliHousing

    data.Predictions = list(lin_gr.lin_reg.predict(lin_gr.some_data_prepared))
    data.Labels = list(lin_gr.some_labels)
    data.rmse_train = lin_gr.rmse_train
    data.mean = lin_gr.rmse_cross_validation.mean()
    data.std = lin_gr.rmse_cross_validation.std()
    data.rmse_test = lin_gr.rmse_test

    predict = md.model_predict_house()
    predict_list = lin_gr.some_data.values.tolist()
    predict_index = lin_gr.some_data.index.tolist()
    predict_labels = lin_gr.some_labels.values.tolist()

    response_object = {
        "Predictions": data.Predictions,
        "Labels": data.Labels,
        "rmse_train": data.rmse_train,
        "mean": data.mean,
        "std": data.std,
        "rmse_test": data.rmse_test,
        "predict_list": predict_list,
        "predict_index" : predict_index,
        "predict_labels" : predict_labels
    }

    return response_object

@router.post("/calihousing/linear-predict-input")
async def getPredict(data: md.model_predict_house):

    newData = md.model_predict_house()

    newData.longitude = data.longitude
    newData.latitude = data.latitude
    newData.housing_median_age = data.housing_median_age
    newData.total_rooms = data.total_rooms
    newData.total_bedrooms = data.total_bedrooms
    newData.population = data.population
    newData.households = data.households
    newData.median_income = data.median_income
    newData.ocean_proximity = data.ocean_proximity

    _data = dict()
    _data.update({'longitude':float(newData.longitude),'latitude':float(newData.latitude),'housing_median_age':float(newData.housing_median_age),'total_rooms':float(newData.total_rooms)})
    _data.update({'total_bedrooms':float(newData.total_bedrooms),'population':float(newData.population),'households':float(newData.households),'median_income':float(newData.median_income)})
    _data.update({'ocean_proximity':newData.ocean_proximity})

    resullt = lin_gr.predict_input_user(_data)

    return resullt.tolist()[0];

# Decision Tree
@router.get("/calihousing/decision-tree-data-sample")
async def getSampleDeCision():
    data =  md.model_CaliHousing

    data.Predictions = list(de_tr_gr.tree_reg.predict(de_tr_gr.some_data_prepared))
    data.Labels = list(de_tr_gr.some_labels)
    data.rmse_train = de_tr_gr.rmse_train
    data.mean = de_tr_gr.rmse_cross_validation.mean()
    data.std = de_tr_gr.rmse_cross_validation.std()
    data.rmse_test = de_tr_gr.rmse_test

    predict = md.model_predict_house()
    predict_list = de_tr_gr.some_data.values.tolist()
    predict_index = de_tr_gr.some_data.index.tolist()
    predict_labels = de_tr_gr.some_labels.values.tolist()

    response_object = {
        "Predictions": data.Predictions,
        "Labels": data.Labels,
        "rmse_train": data.rmse_train,
        "mean": data.mean,
        "std": data.std,
        "rmse_test": data.rmse_test,
        "predict_list": predict_list,
        "predict_index" : predict_index,
        "predict_labels" : predict_labels
    }

    return response_object

@router.post("/calihousing/decision-tree-predict-input")
async def getPredict(data: md.model_predict_house):

    newData = md.model_predict_house()

    newData.longitude = data.longitude
    newData.latitude = data.latitude
    newData.housing_median_age = data.housing_median_age
    newData.total_rooms = data.total_rooms
    newData.total_bedrooms = data.total_bedrooms
    newData.population = data.population
    newData.households = data.households
    newData.median_income = data.median_income
    newData.ocean_proximity = data.ocean_proximity

    _data = dict()
    _data.update({'longitude':float(newData.longitude),'latitude':float(newData.latitude),'housing_median_age':float(newData.housing_median_age),'total_rooms':float(newData.total_rooms)})
    _data.update({'total_bedrooms':float(newData.total_bedrooms),'population':float(newData.population),'households':float(newData.households),'median_income':float(newData.median_income)})
    _data.update({'ocean_proximity':newData.ocean_proximity})

    resullt = de_tr_gr.predict_input_user(_data)

    return resullt.tolist()[0];

@router.get("/calihousing/random-forest-data-sample")
async def getSample():

    data =  md.model_CaliHousing

    data.Predictions = list(rf_gr.forest_reg.predict(rf_gr.some_data_prepared))
    data.Labels = list(rf_gr.some_labels)
    data.rmse_train = rf_gr.rmse_train
    data.mean = rf_gr.rmse_cross_validation.mean()
    data.std = rf_gr.rmse_cross_validation.std()
    data.rmse_test = rf_gr.rmse_test

    predict = md.model_predict_house()
    predict_list = rf_gr.some_data.values.tolist()
    predict_index = rf_gr.some_data.index.tolist()
    predict_labels = rf_gr.some_labels.values.tolist()

    response_object = {
        "Predictions": data.Predictions,
        "Labels": data.Labels,
        "rmse_train": data.rmse_train,
        "mean": data.mean,
        "std": data.std,
        "rmse_test": data.rmse_test,
        "predict_list": predict_list,
        "predict_index" : predict_index,
        "predict_labels" : predict_labels
    }

    return response_object

@router.post("/calihousing/random-forest-predict-input")
async def getPredict(data: md.model_predict_house):

    newData = md.model_predict_house()

    newData.longitude = data.longitude
    newData.latitude = data.latitude
    newData.housing_median_age = data.housing_median_age
    newData.total_rooms = data.total_rooms
    newData.total_bedrooms = data.total_bedrooms
    newData.population = data.population
    newData.households = data.households
    newData.median_income = data.median_income
    newData.ocean_proximity = data.ocean_proximity

    _data = dict()
    _data.update({'longitude':float(newData.longitude),'latitude':float(newData.latitude),'housing_median_age':float(newData.housing_median_age),'total_rooms':float(newData.total_rooms)})
    _data.update({'total_bedrooms':float(newData.total_bedrooms),'population':float(newData.population),'households':float(newData.households),'median_income':float(newData.median_income)})
    _data.update({'ocean_proximity':newData.ocean_proximity})

    resullt = rf_gr.predict_input_user(_data)

    return resullt.tolist()[0];

@router.get("/calihousing/random-forest-grid-search-data-sample")
async def getSample():

    data =  md.model_CaliHousing

    data.Predictions = list(gs_gr.final_model.predict(gs_gr.some_data_prepared))
    data.Labels = list(gs_gr.some_labels)
    data.rmse_train = gs_gr.rmse_train
    data.mean = gs_gr.rmse_cross_validation.mean()
    data.std = gs_gr.rmse_cross_validation.std()
    data.rmse_test = gs_gr.rmse_test

    predict = md.model_predict_house()
    predict_list = gs_gr.some_data.values.tolist()
    predict_index = gs_gr.some_data.index.tolist()
    predict_labels = gs_gr.some_labels.values.tolist()

    response_object = {
        "Predictions": data.Predictions,
        "Labels": data.Labels,
        "rmse_train": data.rmse_train,
        "mean": data.mean,
        "std": data.std,
        "rmse_test": data.rmse_test,
        "predict_list": predict_list,
        "predict_index" : predict_index,
        "predict_labels" : predict_labels
    }

    return response_object

@router.post("/calihousing/random-forest-grid-search-predict-input")
async def getPredict(data: md.model_predict_house):

    newData = md.model_predict_house()

    newData.longitude = data.longitude
    newData.latitude = data.latitude
    newData.housing_median_age = data.housing_median_age
    newData.total_rooms = data.total_rooms
    newData.total_bedrooms = data.total_bedrooms
    newData.population = data.population
    newData.households = data.households
    newData.median_income = data.median_income
    newData.ocean_proximity = data.ocean_proximity

    _data = dict()
    _data.update({'longitude':float(newData.longitude),'latitude':float(newData.latitude),'housing_median_age':float(newData.housing_median_age),'total_rooms':float(newData.total_rooms)})
    _data.update({'total_bedrooms':float(newData.total_bedrooms),'population':float(newData.population),'households':float(newData.households),'median_income':float(newData.median_income)})
    _data.update({'ocean_proximity':newData.ocean_proximity})

    resullt = gs_gr.predict_input_user(_data)

    return resullt.tolist()[0];

@router.get("/calihousing/random-forest-random-search-data-sample")
async def getSample():

    data =  md.model_CaliHousing

    data.Predictions = list(rd_gr.final_model.predict(rd_gr.some_data_prepared))
    data.Labels = list(rd_gr.some_labels)
    data.rmse_train = rd_gr.rmse_train
    data.mean = rd_gr.rmse_cross_validation.mean()
    data.std = rd_gr.rmse_cross_validation.std()
    data.rmse_test = rd_gr.rmse_test

    predict = md.model_predict_house()
    predict_list = rd_gr.some_data.values.tolist()
    predict_index = rd_gr.some_data.index.tolist()
    predict_labels = rd_gr.some_labels.values.tolist()

    response_object = {
        "Predictions": data.Predictions,
        "Labels": data.Labels,
        "rmse_train": data.rmse_train,
        "mean": data.mean,
        "std": data.std,
        "rmse_test": data.rmse_test,
        "predict_list": predict_list,
        "predict_index" : predict_index,
        "predict_labels" : predict_labels
    }

    return response_object

@router.post("/calihousing/random-forest-random-search-predict-input")
async def getPredict(data: md.model_predict_house):

    newData = md.model_predict_house()

    newData.longitude = data.longitude
    newData.latitude = data.latitude
    newData.housing_median_age = data.housing_median_age
    newData.total_rooms = data.total_rooms
    newData.total_bedrooms = data.total_bedrooms
    newData.population = data.population
    newData.households = data.households
    newData.median_income = data.median_income
    newData.ocean_proximity = data.ocean_proximity

    _data = dict()
    _data.update({'longitude':float(newData.longitude),'latitude':float(newData.latitude),'housing_median_age':float(newData.housing_median_age),'total_rooms':float(newData.total_rooms)})
    _data.update({'total_bedrooms':float(newData.total_bedrooms),'population':float(newData.population),'households':float(newData.households),'median_income':float(newData.median_income)})
    _data.update({'ocean_proximity':newData.ocean_proximity})

    resullt = rd_gr.predict_input_user(_data)

    return resullt.tolist()[0];

@router.get("/calihousing/dataset")
async def getDataset():

    dataset = lin_gr.dataset.values.tolist()

    response_object = {
        "Dataset": dataset,
    }

    return response_object
