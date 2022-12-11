from genericpath import sameopenfile
import pathlib
import numpy as np
import pandas as pd
import os, sys
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

from sklearn.model_selection import cross_val_score
from yaml import load

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


def display_scores(scores):
    print("Mean: %.2f" % (scores.mean()))
    print("Standard deviation: %.2f" % (scores.std()))


housing = pd.read_csv('topics/CaliHousing/housing.csv')
dataset = housing.fillna('')

# Them column income_cat dung de chia data
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Chia xong thi delete column income_cat
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

housing_num = housing.drop("ocean_proximity", axis=1)

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)

def store_model(model, model_name = ""):
    if model_name == "": 
        model_name = type(model).__name__
    joblib.dump(model,'topics/CaliHousing/models/' + model_name + '_model.pkl')
def load_model(model_name):
    # Load objects into memory
    #del model
    model = joblib.load('topics/CaliHousing/models/' + model_name + '_model.pkl')
    #print(model)
    return model

#store_model(lin_reg)

# Training
#lin_reg = LinearRegression()
#lin_reg.fit(housing_prepared, housing_labels)

# load model da trainning rồi
# Prediction
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
'''print(some_data.values.tolist())
print(some_data.index.tolist())'''

lin_reg = load_model("LinearRegression")

# Prediction 5 samples 
Predictions = lin_reg.predict(some_data_prepared)
Labels = list(some_labels)

'''print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))
print('\n')'''


# Tính sai số bình phương trung bình trên tập dữ liệu huấn luyện
housing_predictions = lin_reg.predict(housing_prepared)
mse_train = mean_squared_error(housing_labels, housing_predictions)
rmse_train = np.sqrt(mse_train)

# Tính sai số bình phương trung bình trên tập dữ liệu kiểm định chéo (cross-validation) 
scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_cross_validation = np.sqrt(-scores)
#display_scores(rmse_cross_validation)

# Tính sai số bình phương trung bình trên tập dữ liệu kiểm tra (test)
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)
y_predictions = lin_reg.predict(X_test_prepared)
mse_test = mean_squared_error(y_test, y_predictions)
rmse_test = np.sqrt(mse_test)

def predict_input_user(data):
    row_label = [1]
    sample = pd.DataFrame(data=data,index=row_label)
    sample_prepared = full_pipeline.transform(sample)
    
    return lin_reg.predict(sample_prepared)

    


