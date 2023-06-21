import pandas as pd
from tabu_search import run_ts_fs

def feature_selection(runs):
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

# feature selection

feature_selection(10)

