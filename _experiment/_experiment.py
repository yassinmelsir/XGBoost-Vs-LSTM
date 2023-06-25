from analysis import *
from figures import figure_one, figure_two
import numpy as np

# xg_features, lstm_features = fs(10) # feature selection

# Possible features: 2-28
feature_solution_1 = [9, 10, 11]
feature_solution_2 = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]


# xg_w_nofs() # xg with no feature selection

# lstm_w_nofs() # lstm with no feature selection

xg_w_fs(feature_solution_2) # xg with feature selection

lstm_w_fs(feature_solution_2) # lstm with feature selection

# curve file function

# predict curve file function

# function figure curved file v normal file 

figure_two() # figures

# get working

# change evaluation function


