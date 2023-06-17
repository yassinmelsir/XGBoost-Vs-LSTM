import pandas as pd
import numpy as np

df = pd.read_csv('/Users/yme/code/AppliedAI/summativeassessment/data/full_dataset.csv').fillna(0)

X, y = df[df.columns.difference(['ISO','Country'])], df[['Country','Year','Total']]

cats = X.select_dtypes(exclude=np.number).columns.to_list()
for col in cats: X[col] = X[col].astype('category')

