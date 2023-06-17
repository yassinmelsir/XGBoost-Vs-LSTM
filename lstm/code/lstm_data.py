import pandas as pd
from lstm_functions import encode
from sklearn.preprocessing import MinMaxScaler

#input UK renewables
#out CO2 emissions
df = pd.read_csv('/Users/yme/code/AppliedAI/summativeassessment/data/full_dataset.csv').fillna(0)

#input UK renewables
#out CO2 emissions
# X = df[['Country','Year','Fraction from renewable sources and waste']].fillna(0)
# y = df[['Country','Year','Total']].fillna(0)

#input all data 
#output CO2 Emissions

X = df[df.columns.difference(['ISO','Total'])]
y = df[['Country','Year','Total']]

X, y = encode(df,X,y)

#normalize data

#unnormalize data







