import xgboost as xgb
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def runtraining(df):
    #split features and target
    X, y = df.drop('Total', axis=1), df[['Country','Year','Total']]

    # covert cols to proper type
    cats = X.select_dtypes(exclude=np.number).columns.to_list()
    for col in cats: X[col] = X[col].astype('category')

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

    # create Dataframe for country, real total, predicted total
    preds = y_test_org.join(pd.DataFrame(data={'Predicted Total': preds})).fillna('r')
    filtered_preds = preds[preds['Predicted Total'] != 'r']

    # print result and error
    # print(filtered_preds)
    print(f"RMSE of the base model: {rmse:.3f}")

    model.save_model('/Users/yme/code/AppliedAI/summativeassessment/xg/model.json')