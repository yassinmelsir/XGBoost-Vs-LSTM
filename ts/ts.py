import random
from lstm.lstm import lstm_init, lstm_predict, getlstmdata
from xg.xg import xg_init, xg_predict, getxgdata

def run_ts_fs(model, runs):
    evaluation_function = lstm_evaluate if model=='lsmtm' else xg_evaluate
    return TabuSearch(evaluation_function, runs) 

def lstm_evaluate(s):
    X, y = getlstmdata()
    fts_to_drop = [fts for fts in range(2,28) if fts not in s] 
    for fts in fts_to_drop: X = X.drop(X.columns[fts], axis=1)
    lstm_init(X,y)
    rmse = lstm_predict(X,y)
    return rmse, s

def xg_evaluate(s):
    X, y = getxgdata()
    fts_to_drop = [fts for fts in range(2,28) if fts not in s] 
    for fts in fts_to_drop: X = X.drop(X.columns[fts], axis=1)
    xg_init(X,y)
    rmse = xg_predict(X,y)
    return rmse, s

def TabuSearch(evaluation_function, runs):
# tabu list to conserve number of evaluation runs
    i_solution = list(range(2,28))
    F_history, T_list = [evaluation_function(i_solution)], [i_solution]
    # canadidate solution variable unused due to modified neighboor generation function
    # final solution unused to minimize the computationally expensive evaluation function
    # f_solution, c_solution = i_solution.copy(), i_solution.copy()
    itrns = 0
    while itrns < runs:
        #neighbourhood generated randomly for entire set of solutions rather than in neighborhood of candidate solution
        nghbrhd = [random.randint(2,28) for _ in range(0,5) for __ in range(random.randint(0,28))]
        #evaluate neighbourhood solutions
        evaluations = [evaluation_function(s) for s in nghbrhd]
        #find best solution in neighbourhood
        prime_score, prime_solution = max(evaluations, key=lambda x: x[0])
        #aspiration criteria
        if prime_score >= F_history[-1][0]: F_history.append([prime_score, prime_solution])
        # f_solution, c_solution = prime_solution.copy(), prime_solution.copy()
        #set new candidate solution and update Tabu List
        T_list.append(prime_solution)
        itrns+=1
    return F_history

