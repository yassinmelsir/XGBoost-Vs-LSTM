import random
import math
from general_functions import get_data
from lstm_neural_network import lstm_init, lstm_predict
from gradient_boosted_trees import xg_init, xg_predict

#Tabu Search

def run_ts_fs(model, runs):
    evaluation_function = lstm_evaluate if model=='lstm' else xg_evaluate
    # def evaluate(array):
    #     return sum(array), array
    # evaluation_function = evaluate
    return TabuSearch(evaluation_function, runs) 

def lstm_evaluate(s):
    X, y = get_data('full')
    fts_to_drop = [fts for fts in range(2,28) if fts not in s] 
    X = X.drop(X.columns[fts_to_drop], axis=1)
    lstm_init(X,y)
    y, preds, rmse = lstm_predict(X,y)
    s = sorted(s)
    return rmse, s

def xg_evaluate(s):
    X, y = get_data('full')
    fts_to_drop = [fts for fts in range(2,28) if fts not in s]
    X = X.drop(X.columns[fts_to_drop], axis=1)
    xg_init(X,y)
    preds, rmse = xg_predict(X, y)
    s = sorted(s)
    return rmse, s

def TabuSearch(evaluation_function, runs):
    itrns = 0
# tabu list to conserve number of evaluation runs
    i_solution = list(range(2,28))
    F_history, T_list = [evaluation_function(i_solution)], [i_solution]
    BestIterations = [itrns]
    # final solution unused to minimize the computationally expensive evaluation function
    f_solution, c_solution = i_solution.copy(), i_solution.copy()
    while itrns < runs:
        itrns+=1
        #neighborhood generation heuristic: retain up to half of candidate solution and randomly generate the other half, while allowing for shorter or longer solutions
        nghbrhd = [list(set(random.sample(c_solution, random.randint(2, 14)) + random.sample(range(2, 29), random.randint(2, 13)))) for _ in range(5)]
        #evaluate neighbourhood solutions
        evaluations =  [evaluation_function(s) for s in nghbrhd if s not in T_list]
        #find best solution in neighbourhood
        prime_score, prime_solution = min(evaluations, key=lambda x: x[0])
        #aspiration criteria
        if prime_score <= F_history[-1][0]: F_history.append([prime_score, prime_solution,itrns]); BestIterations.append(itrns)
        # f_solution, c_solution = prime_solution.copy(), prime_solution.copy() 
        #set new candidate solution and update Tabu List
        # append all evaluations
        T_list += nghbrhd
    results = [[element[0],element[1],len(element[1]),BestIterations[idx]] for idx, element in enumerate(F_history)]
    return results

