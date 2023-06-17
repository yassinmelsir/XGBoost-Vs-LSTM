import pandas as pd
from xg_functions import *


df = pd.read_csv('/Users/yme/code/AppliedAI/summativeassessment/data/full_dataset.csv').fillna(0)

X, y = df[df.columns.difference(['ISO','Country','Coal'])], df[['Country','Year','Total','Coal']]


categorizecols(X)
    




xg_init(X, y)

preds = xg_predict(X, y)
predictionsfigure(preds, X, y)
