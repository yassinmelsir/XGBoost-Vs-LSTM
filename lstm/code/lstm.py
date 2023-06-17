import pandas as pd
from lstm_functions import *
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('/Users/yme/code/AppliedAI/summativeassessment/data/full_dataset.csv').fillna(0)

X, y = df[df.columns.difference(['ISO','Total'])], df[['Country','Year','Total']]














X, y = encode(df,X,y)

lstm1 = lstm_init(X,y)

data_predict, dataY_plot = lstm_predict(X,y)
predictionsfigure(data_predict,dataY_plot)



