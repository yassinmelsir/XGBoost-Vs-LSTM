import pandas as pd
import matplotlib.pyplot as plt

def plotrenewables(df):
    years = df['Year']
    renewables = df['Fraction from renewable sources and waste']

    plt.plot(years, renewables, label='Renewables')

    plt.xlabel('Year')
    plt.ylabel('Fraction of Energy from Renewable Sources and Waste')
    plt.title('Renewable Energy Market Cap Over Time')
    plt.legend()
    plt.show()

def plotemissions(df):
    years = df['Year']
    emissions = df['Total']

    plt.plot(years, emissions, label='Emissions')

    plt.xlabel('Year')
    plt.ylabel('CO2 Emissions')
    plt.title('CO2 Emissions Over Time')
    plt.legend()
    plt.show()