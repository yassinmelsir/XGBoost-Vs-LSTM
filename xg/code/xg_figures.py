import matplotlib.pyplot as plt

def createfigure(preds, X, y):
    predicted_year= X['Year'].to_numpy()
    predicted_emissions = preds

    actual_year = X['Year'].to_numpy()
    actual_emissions = y['Total'].to_numpy()
    
    plt.figure(figsize=(10,6))
    plt.plot(predicted_year,predicted_emissions,label='Predicted')
    plt.plot(actual_year,actual_emissions,label='Actual')
    plt.xlabel('Year')
    plt.ylabel('Emissions')
    plt.title('Actual vs Predicted Emissions per Year')
    plt.legend()
    plt.show()