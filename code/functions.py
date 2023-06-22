import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def get_data(dataset):
    df = pd.read_csv('/Users/yme/code/AppliedAI/summativeassessment/data/full_dataset.csv').fillna(0)

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

# def multiple_runs_figure():
#     predicted_year= preds['Year'].to_numpy()
#     predicted_emissions = preds

#     actual_year = input_X['Year'].to_numpy()
#     actual_emissions = input_y['Total'].to_numpy()
    
#     plt.figure(figsize=(10,6))
#     for pred in preds:

#         predicted_year= X['Year'].to_numpy()
#         predicted_emissions = pred
#         plt.plot(predicted_year,predicted_emissions,label='Predicted')
    


#     plt.plot(actual_year,actual_emissions,label='Actual')
#     plt.xlabel('Year')
#     plt.ylabel('Emissions')
#     plt.title('Actual vs Predicted Emissions per Year')

#     #performance plot

#     plt.legend()
#     plt.show()

# def performance_figure(preds, X, y):
#     predicted_emissions = preds
#     actual_emissions = y['Total'].to_numpy()
    
#     plt.figure(figsize=(10,6))
#     plt.plot(predicted_emissions,label='Predicted')
#     plt.plot(actual_emissions,label='Actual')
#     plt.xlabel('Year')
#     plt.ylabel('Emissions')
#     plt.title('Actual vs Predicted Emissions per Year')
#     plt.legend()
#     plt.show()

# def predictions_figure(preds, X, y):
#     predicted_year= X['Year'].to_numpy()
#     predicted_emissions = preds

#     actual_year = X['Year'].to_numpy()
#     actual_emissions = y['Total'].to_numpy()
    
#     plt.figure(figsize=(10,6))
#     plt.plot(predicted_year,predicted_emissions,label='Predicted')
#     plt.plot(actual_year,actual_emissions,label='Actual')
#     plt.xlabel('Year')
#     plt.ylabel('Emissions')
#     plt.title('Actual vs Predicted Emissions per Year')
#     plt.legend()
#     plt.show()

# def plus30years(df): 
#     newDf = df[0:0].copy()
#     countries = df['Country'].unique()
#     for country in countries:
#         oldCountry = df[(df['Country'] == country) & (df['Year'] >= 1990)].copy()
#         newCountry = oldCountry[(oldCountry['Year'] >= 1990)].copy()
#         newCountry['Year'] = newCountry['Year'] + 30
        
#         for column in newDf.columns.values:
#             if (column != 'Country') & (column != 'Year'): newCountry[column] = float(0)
#         newDfSlice = pd.concat([oldCountry,newCountry])
#         newDf = pd.concat([newDf,newDfSlice]).sort_values(by=['Country', 'Year'], ascending=True)
#     return newDf

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

# def next35Years(df): 
#     newDf = df[0:0].copy()
#     countries = df['Country'].unique()
#     for country in countries:
#         oldCountry = df[(df['Country'] == country) & (df['Year'] >= 1990)].copy()
#         newCountry = oldCountry.copy()
#         newCountry['Year'] = newCountry['Year'] + 35
#         for column in newDf.columns.values:
#             if (column != 'Country') & (column != 'Year'): newCountry[column] = float(0)
#         newDf = pd.concat([newDf,newCountry])
#     return newDf

# def country(df, label): 
#     country = df[((df['Country'] == label) & (df['Year'] >= 1990))]
#     country['Year'] = country['Year'] + 32
#     for column in df.columns.values:
#         if (column != 'Country') & (column != 'Year'): country[column] = float(0)
#     return country

# def getxgdata():
#     df = pd.read_csv('/Users/yme/code/AppliedAI/summativeassessment/data/full_dataset.csv').fillna(0)

#     X, y = df.drop(columns=['ISO','Total']), df[['Country','Year','Total']]
    
#     categorizecols(X)
#     return X, y

# def categorizecols(X):
#     cats = X.select_dtypes(exclude=np.number).columns.to_list()
#     for col in cats: X[col] = X[col].astype('category')