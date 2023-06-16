import pandas as pd
def plus30years(df): 
    newDf = df[0:0].copy()
    countries = df['Country'].unique()
    for country in countries:
        oldCountry = df[(df['Country'] == country) & (df['Year'] >= 1990)].copy()
        newCountry = oldCountry[(oldCountry['Year'] >= 1990)].copy()
        newCountry['Year'] = newCountry['Year'] + 30
        
        for column in newDf.columns.values:
            if (column != 'Country') & (column != 'Year'): newCountry[column] = float(0)
        newDfSlice = pd.concat([oldCountry,newCountry])
        newDf = pd.concat([newDf,newDfSlice]).sort_values(by=['Country', 'Year'], ascending=True)
    return newDf

def lastsixtyyears(df): 
    newDf = df[0:0].copy()
    countries = df['Country'].unique()
    for country in countries:
        oldCountry = df[(df['Country'] == country) & (df['Year'] >= 1990)].copy()
        newCountry = oldCountry.copy()
        newCountry['Year'] = newCountry['Year'] + 32
        for column in newDf.columns.values:
            if (column != 'Country') & (column != 'Year'): newCountry[column] = float(0)
        newDfSlice = pd.concat([oldCountry,newCountry])
        newDf = pd.concat([newDf,newDfSlice])
    return newDf

def next35Years(df): 
    newDf = df[0:0].copy()
    countries = df['Country'].unique()
    for country in countries:
        oldCountry = df[(df['Country'] == country) & (df['Year'] >= 1990)].copy()
        newCountry = oldCountry.copy()
        newCountry['Year'] = newCountry['Year'] + 35
        for column in newDf.columns.values:
            if (column != 'Country') & (column != 'Year'): newCountry[column] = float(0)
        newDf = pd.concat([newDf,newCountry])
    return newDf

def country(df, label): 
    country = df[((df['Country'] == label) & (df['Year'] >= 1990))]
    country['Year'] = country['Year'] + 32
    for column in df.columns.values:
        if (column != 'Country') & (column != 'Year'): country[column] = float(0)
    return country