import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def run_xg():
    X, y = getxgdata()
    xg_init(X, y)
    preds, rmse = xg_predict(X, y)
    predictionsfigure(preds, X, y)

def getxgdata():
    df = pd.read_csv('/Users/yme/code/AppliedAI/summativeassessment/data/full_dataset.csv').fillna(0)

    X, y = df.drop(columns=['ISO','Total']), df[['Country','Year','Total']]
    
    categorizecols(X)
    return X, y

def categorizecols(X):
    cats = X.select_dtypes(exclude=np.number).columns.to_list()
    for col in cats: X[col] = X[col].astype('category')

def plus30years(df): 
    newDf = df[0:0].copy()
    countries = df['Country'].unique()
    for country in countries:
        oldCountry = df[(df['Country'] == country) & (df['Year'] >= 1990)].copy()
        newCountry = oldCountry[(oldCountry['Year'] >= 1990)].copy()
        newCountry['Year'] = newCountry['Year'] + 30
        
        for column in newDf.columns.values:
            if (column != 'Country') & (column != 'Year'): newCountry[column] = float(0)
        newDfSlice = pd.concat([oldCountry,newCountry])
        newDf = pd.concat([newDf,newDfSlice]).sort_values(by=['Country', 'Year'], ascending=True)
    return newDf

def lastsixtyyears(df): 
    newDf = df[0:0].copy()
    countries = df['Country'].unique()
    for country in countries:
        oldCountry = df[(df['Country'] == country) & (df['Year'] >= 1990)].copy()
        newCountry = oldCountry.copy()
        newCountry['Year'] = newCountry['Year'] + 32
        for column in newDf.columns.values:
            if (column != 'Country') & (column != 'Year'): newCountry[column] = float(0)
        newDfSlice = pd.concat([oldCountry,newCountry])
        newDf = pd.concat([newDf,newDfSlice])
    return newDf

def next35Years(df): 
    newDf = df[0:0].copy()
    countries = df['Country'].unique()
    for country in countries:
        oldCountry = df[(df['Country'] == country) & (df['Year'] >= 1990)].copy()
        newCountry = oldCountry.copy()
        newCountry['Year'] = newCountry['Year'] + 35
        for column in newDf.columns.values:
            if (column != 'Country') & (column != 'Year'): newCountry[column] = float(0)
        newDf = pd.concat([newDf,newCountry])
    return newDf

def country(df, label): 
    country = df[((df['Country'] == label) & (df['Year'] >= 1990))]
    country['Year'] = country['Year'] + 32
    for column in df.columns.values:
        if (column != 'Country') & (column != 'Year'): country[column] = float(0)
    return country

def performancefigure(preds, X, y):
    predicted_emissions = preds
    actual_emissions = y['Total'].to_numpy()
    
    plt.figure(figsize=(10,6))
    plt.plot(predicted_emissions,label='Predicted')
    plt.plot(actual_emissions,label='Actual')
    plt.xlabel('Year')
    plt.ylabel('Emissions')
    plt.title('Actual vs Predicted Emissions per Year')
    plt.legend()
    plt.show()

def predictionsfigure(preds, X, y):
    predicted_year= X['Year'].to_numpy()
    predicted_emissions = preds

    actual_year = X['Year'].to_numpy()
    actual_emissions = y['Total'].to_numpy()
    
    plt.figure(figsize=(10,6))
    plt.plot(predicted_year,predicted_emissions,label='Predicted')
    plt.plot(actual_year,actual_emissions,label='Actual')
    plt.xlabel('Year')
    plt.ylabel('Emissions')
    plt.title('Actual vs Predicted Emissions per Year')
    plt.legend()
    plt.show()

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
    print(f"RMSE of the base model: {rmse:.3f}")

    model.save_model('/Users/yme/code/AppliedAI/summativeassessment/xg/xg_model.json')

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
    model.load_model('/Users/yme/code/AppliedAI/summativeassessment/xg/xg_model.json')

    # #define prediction and calculate error
    preds = model.predict(dtest_reg)
    rmse = mean_squared_error(y_test, preds, squared=False)

    # # create Dataframe for country, real total, predicted total
    print(f"RMSE of the base model: {rmse:.3f}")
    # print result and error
    return preds, rmse

run_xg()