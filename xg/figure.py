import matplotlib.pyplot as plt

def createfigure(df):
    years = df['Year']
    actual = df['Total']
    predictions = df['Predicted Total']

    plt.plot(years, actual, label='Actual Emissions')
    plt.plot(years, predictions, label='Predicted Emissions')

    plt.xlabel('Year')
    plt.ylabel('CO2 Emissions')
    plt.title('CO2 Emissions Over Time')
    plt.legend()
    plt.show()