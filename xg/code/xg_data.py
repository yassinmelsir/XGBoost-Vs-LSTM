import pandas as pd
import numpy as np
from xg_functions import categorizecols

df = pd.read_csv('/Users/yme/code/AppliedAI/summativeassessment/data/full_dataset.csv').fillna(0)

X, y = df[df.columns.difference(['ISO','Country','Coal'])], df[['Country','Year','Total','Coal']]

categorizecols(X)
    

