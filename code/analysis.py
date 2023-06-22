import pandas as pd
from functions import decode, get_data
from gradient_boosted_trees import run_xg
from lstm_neural_network import run_lstm
from tabu_search import run_ts_fs

# feature selection
def fs(runs):
    # number of runs for the feature selection 
    # #run feature selection for each model
    xg_feature_selection = run_ts_fs('xg',runs)
    lstm_feature_selection = run_ts_fs('lstm',runs)
    # # save feature selection results to dataframe 
    feature_selection_results = [['xg',result[0],result[1],result[2],result[2]] for result in xg_feature_selection]
    for result in lstm_feature_selection: feature_selection_results.append(['lstm',result[0],result[1],result[2],result[3]])
    df = pd.DataFrame(data=feature_selection_results, columns=['model','rmse','solution','length of solution','iterations'])
    # write dataframe to csv
    df.to_csv('/Users/yme/code/AppliedAI/summativeassessment/data/feature_selection_results.csv', index=False)
    return xg_feature_selection[2],lstm_feature_selection[2]

# xg boost without feature selection with normal dataset
def xg_w_nofs():
    rmse, preds, y = run_xg('full')
    df = pd.read_csv('/Users/yme/code/AppliedAI/summativeassessment/data/full_dataset.csv').fillna(0)
    y = decode(df,y)
    results = y.copy()
    results['Predicted Emissions'] = preds
    # preds is numpy
    # y is dataframe
    results.to_csv(f'/Users/yme/code/AppliedAI/summativeassessment/data/xg_nofs_results.csv', index=False)
# lstm without feature selection with normal dataset
def lstm_w_nofs():
    dataY_plot, data_predict, rmse = run_lstm('full')
    X, y = get_data('full')
    df = pd.read_csv('/Users/yme/code/AppliedAI/summativeassessment/data/full_dataset.csv').fillna(0)
    y = decode(df,y)
    results = y.copy()
    results['Predicted Emissions'] = data_predict[:,2]
    # preds is numpy
    # y is dataframe
    results.to_csv(f'/Users/yme/code/AppliedAI/summativeassessment/data/lstm_nofs_results.csv', index=False)
# xg boost with feature selection with normal dataset
def xg_w_fs(column_indexes):
    rmse, preds, y = run_xg(column_indexes)
    df = pd.read_csv('/Users/yme/code/AppliedAI/summativeassessment/data/full_dataset.csv').fillna(0)
    y = decode(df,y)
    results = y.copy()
    results['Predicted Emissions'] = preds
    # preds is numpy
    # y is dataframe
    results.to_csv(f'/Users/yme/code/AppliedAI/summativeassessment/data/xg_fs_results.csv', index=False)
# lstm with feature selection with normal dataset
def lstm_w_fs(column_indexes):
    dataY_plot, data_predict, rmse = run_lstm(column_indexes)
    X, y = get_data('full')
    df = pd.read_csv('/Users/yme/code/AppliedAI/summativeassessment/data/full_dataset.csv').fillna(0)
    y = decode(df,y)
    results = y.copy()
    results['Predicted Emissions'] = data_predict[:,2]
    # preds is numpy
    # y is dataframe
    results.to_csv('/Users/yme/code/AppliedAI/summativeassessment/data/lstm_fs_results.csv', index=False)



