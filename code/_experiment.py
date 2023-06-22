from analysis import *
from figures import figure_one, figure_two

xg_best_features, lstm_best_features = fs(10) # feature selection

features = [3,4,5]

xg_w_nofs() # xg with no feature selection

lstm_w_nofs() # lstm with no feature selection

xg_w_fs(xg_best_features) # xg with feature selection

lstm_w_fs(lstm_best_features) # lstm with feature selection

figure_two()
