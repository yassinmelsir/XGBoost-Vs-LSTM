from analysis_functions import *
from figure_functions import figure_for_file,multi_figure

#Play with the results produced by changing the curve multiplier and features selection choice.
#All results stored in data/results. A single file can be plotted as a figure or with four other files.
#Comment out functions to run individually

curve = 2 # multiplier for renewable energy data. Changes amount of renewables.
xg_features_solution = 2 # set of features to use. choose 1 or 2
lstm_features_solution = 2 # set of features to use. choose 1 or 2

xg_w_nofs() # xg without feature selection. results stores in csv under results directory
lstm_w_nofs() # lstm without feature selection. results stores in csv under results directory

xg_w_fs(xg_features_solution,curve) # xgboost with feature selection. results stores in csv under results directory
lstm_w_fs(lstm_features_solution,curve) # lstm with feature selection. results stores in csv under results directory

figure_for_file('lstm_nofs_results') # file name to graph.  don't add .csv
multi_figure('xg_nofs_results','lstm_nofs_results','xg_fs1_c1','lstm_fs1_c1') # file names to graph. don't add .csv

# xg_features, lstm_features = fs(10) # feature selection


