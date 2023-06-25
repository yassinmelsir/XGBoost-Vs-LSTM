from analysis_functions import *
from figure_functions import figure_one, figure_two
import numpy as np

# xg_features, lstm_features = fs(10) # feature selection. Does it still work?


xg_w_nofs() # xg without feature selection
lstm_w_nofs() # lstm without feature selection

# input True for feature selection solution 1, input False for feature selection input 2
curve = 1
xg_w_fs(fs_solution_one=True,curve=curve) # xgboost with feature selection
lstm_w_fs(fs_solution_one=True,curve=curve) # lstm with feature selection

figure_two() # figures for 


