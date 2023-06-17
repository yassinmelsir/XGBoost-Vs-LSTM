from xg_functions import *
from xg_data import X, y
from xg_figures import createfigure
import xgboost as xgb


#experiment with running a multiple output
xg_init(X, y)
preds = xg_predict(X, y)
createfigure(preds, X, y)
