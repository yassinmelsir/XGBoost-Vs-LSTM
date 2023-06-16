from createpreddf import lastsixtyyears, next35Years, country, plus30years
from runtraining import runtraining
from runexperiment import runexperiment
from figure import createfigure
from df import df
import xgboost as xgb

#experiment with running a multiple output


def experiment(df):
    ndf = plus30years(df)
    preds = runexperiment(ndf)[0]
    createfigure(preds)
    print(preds.head())

runtraining(df)
experiment(df)
