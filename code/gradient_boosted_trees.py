import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from general_functions import get_data, curve_data

#XGBoost Extreme Gradient Boosted Trees

def run_xg(dataset,curve=1,filename=''):
    X, y = get_data(dataset)
    xg_init(X, y)
    if filename != '': X, y = get_data(dataset,filename)
    X = curve_data(X,curve)
    preds, rmse = xg_predict(X, y)
    return rmse, preds, y

def xg_init(X, y):
    #split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    #conserve countries for each row of test array, 
    y_test_org = y_test

    #create a 1d output array
    y_train = y_train.drop('Country',axis=1).drop('Year',axis=1)
    y_test = y_test.drop('Country',axis=1).drop('Year',axis=1)

    # # Create regression matrices
    dtrain_reg = xgb.DMatrix(X_train, y_train, enable_categorical=True)
    dtest_reg = xgb.DMatrix(X_test, y_test, enable_categorical=True)

    # # Define hyperparameters
    params = {"objective": "reg:squarederror", "tree_method": "hist"}
    evals = [(dtrain_reg, "train"), (dtest_reg, "validation")]

    # define model

    model = xgb.train(
    params=params,
    dtrain=dtrain_reg,
    num_boost_round=10000,
    evals=evals,
    verbose_eval=500,
    early_stopping_rounds=100
    )

    # #define prediction and calculate error
    preds = model.predict(dtest_reg)
    rmse = mean_squared_error(y_test, preds, squared=False)

    # print result and error
    print(f"Initial RMSE: {rmse:.3f}")

    model.save_model('/Users/yme/code/AppliedAI/summativeassessment/models/xg.json')

    return preds

def xg_predict(X,y):
    X_test, y_test = X, y
    # conserve countries for each row of test array, 
    y_test_org = y_test

    #create a 1d output array
    y_test = y_test.drop('Country',axis=1).drop('Year',axis=1)

    # # Create regression matrices
    dtest_reg = xgb.DMatrix(X_test, y_test, enable_categorical=True)

    # define model
    model = xgb.Booster()
    model.load_model('/Users/yme/code/AppliedAI/summativeassessment/models/xg.json')

    # #define prediction and calculate error
    preds = model.predict(dtest_reg)
    rmse = mean_squared_error(y_test, preds, squared=False)

    # # create Dataframe for country, real total, predicted total
    print(f"XG Prediction RMSE: {rmse:.3f}")
    # print result and error
    return preds, rmse
