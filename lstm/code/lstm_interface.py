from lstm_data import X, y
from lstm_functions import *


lstm1 = lstm_init(X,y)

data_predict, dataY_plot = lstm_predict(X,y)

predictionsfigure(data_predict,dataY_plot)