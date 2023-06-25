from analysis_functions import *
from figure_functions import figure_one, figure_two
import numpy as np

# xg_features, lstm_features = fs(10) # feature selection


xg_w_nofs() # xg without feature selection
lstm_w_nofs() # lstm without feature selection

# input True for feature selection solution 1, input False for feature selection input 2
xg_w_fs(False) # xgboost with feature selection
lstm_w_fs(False) # lstm with feature selection

# curve file function

# predict curve file function

# function figure curved file v normal file 

figure_two() # figures for 


