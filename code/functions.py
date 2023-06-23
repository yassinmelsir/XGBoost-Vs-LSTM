import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def get_data(dataset,filename=''):
    df = pd.read_csv('/Users/yme/code/AppliedAI/summativeassessment/data/training/full_dataset.csv').fillna(0)
    if filename != '': df = pd.read_csv(f'/Users/yme/code/AppliedAI/summativeassessment/data/{filename}.csv').fillna(0)
    if dataset != 'full': df = df[(df['Country']=='United Kingdom')]
    y = df[['Country','Year','Total']] 
    X = df.drop(columns=['ISO','Total']) 
    if dataset != 'full': X = X.iloc[:,[0,1] + dataset]
    
    X, y = encode(df,X,y)

    return X, y

def encode(df,X,y):

    def encode_country(country):
        return label_encoder.transform([country])[0]
    label_encoder = LabelEncoder()
    label_encoder.fit(df['Country'])
    X['Country'] = X['Country'].apply(encode_country)
    y['Country'] = y['Country'].apply(encode_country)

    return X, y

def decode(df,y):

    def decode_country(country):
        return label_encoder.inverse_transform([country])[0]
    label_encoder = LabelEncoder()
    label_encoder.fit(df['Country'])
    # X['Country'] = X['Country'].apply(decode_country)
    y['Country'] = y['Country'].apply(decode_country)

    return y

def extended_dataset(): 
    df = pd.read_csv('/Users/yme/code/AppliedAI/summativeassessment/data/training/full_dataset.csv').fillna(0)
    newDf = df.copy()
    countries = df['Country'].unique()
    for country in countries:
        oldCountry = df[(df['Country'] == country) & (df['Year'] >= 1990)].copy()
        newCountry = oldCountry.copy()
        newCountry['Year'] = newCountry['Year'] + 32
        for column in newDf.columns.values:
            if (column != 'Country') & (column != 'Year'): newCountry[column] = float(0)
        # newDfSlice = pd.concat([oldCountry,newCountry])
        newDf = pd.concat([newDf,newCountry])
    return newDf

def gen_tool_dataset(filename,country='',random_energy_data=False):
    df = pd.read_csv('/Users/yme/code/AppliedAI/summativeassessment/data/training/full_dataset.csv').fillna(0)
    cntry = 'United Kingdom'if country == '' else country
    df = df[(df['Country'] == f'{cntry}')& (df['Year'] >= 1990)]
    if random_energy_data:
        print('modify renewable data')
    df.to_csv(f'/Users/yme/code/AppliedAI/summativeassessment/data/{filename}.csv',index=False)

