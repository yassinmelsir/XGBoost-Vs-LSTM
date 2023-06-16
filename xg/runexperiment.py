import xgboost as xgb
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def runexperiment(df):
    #split features and target
    X, y = df.drop('Total', axis=1), df[['Country','Year','Total']]

    # covert cols to proper type
    cats = X.select_dtypes(exclude=np.number).columns.to_list()
    for col in cats: X[col] = X[col].astype('category')

    #split data
    #enforce that one of every country is in each data set
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    X_test, y_test = X, y
    # conserve countries for each row of test array, 
    y_test_org = y_test

    #create a 1d output array
    y_test = y_test.drop('Country',axis=1).drop('Year',axis=1)

    # # Create regression matrices
    dtest_reg = xgb.DMatrix(X_test, y_test, enable_categorical=True)

    # define model
    model = xgb.Booster()
    model.load_model('/Users/yme/code/AppliedAI/summativeassessment/xg/model.json')

    # #define prediction and calculate error
    preds = model.predict(dtest_reg)
    print(preds)
    rmse = mean_squared_error(y_test, preds, squared=False)

    # # create Dataframe for country, real total, predicted total
    preds = y_test_org.join(pd.DataFrame(data={'Predicted Total': preds}))
    print(preds)
    print(f"RMSE of the base model: {rmse:.3f}")
    # print result and error
    return preds, rmse
    