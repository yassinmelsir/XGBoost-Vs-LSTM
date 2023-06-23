from analysis import *
from figures import figure_one, figure_two

# xg_features, lstm_features = fs(10) # feature selection

xg_features = [2, 3, 4, 5, 8, 9, 11, 12, 13, 14, 15, 17, 19, 20, 25]
lstm_features = [2, 4, 6, 9, 10, 16, 17, 18, 19, 23, 24, 28]

xg_w_nofs() # xg with no feature selection

lstm_w_nofs() # lstm with no feature selection

xg_w_fs(xg_features) # xg with feature selection

lstm_w_fs(lstm_features) # lstm with feature selection

figure_two() # figures

# get working

# change evaluation function


