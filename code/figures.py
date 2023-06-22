import pandas as pd
import matplotlib.pyplot as plt

def figure_one():
    # Read the CSV files
    file1 = pd.read_csv('/Users/yme/code/AppliedAI/summativeassessment/data/lstm_nofs_results.csv')
    file2 = pd.read_csv('/Users/yme/code/AppliedAI/summativeassessment/data/xg_nofs_results.csv')
    file3 = pd.read_csv('/Users/yme/code/AppliedAI/summativeassessment/data/lstm_fs_results.csv')
    file4 = pd.read_csv('/Users/yme/code/AppliedAI/summativeassessment/data/xg_fs_results.csv')

    # Plotting actual and predicted emissions for each file
    plt.figure(figsize=(10, 6))

    # File 1
    plt.plot(file1['Year'], file1['Total'], label='File 1 - Actual')
    plt.plot(file1['Year'], file1['Predicted Emissions'], label='lstm nofs - Predicted')

    # File 2
    # plt.plot(file2['Year'], file2['Total'], label='File 2 - Actual')
    plt.plot(file2['Year'], file2['Predicted Emissions'], label='xg no fs - Predicted')

    # File 3
    # plt.plot(file3['Year'], file3['Total'], label='File 3 - Actual')
    plt.plot(file3['Year'], file3['Predicted Emissions'], label='lstm fs - Predicted')

    # File 4
    # plt.plot(file4['Year'], file4['Total'], label='File 4 - Actual')
    plt.plot(file4['Year'], file4['Predicted Emissions'], label='xg fs - Predicted')

    # Set labels and title
    plt.xlabel('Year')
    plt.ylabel('Emissions')
    plt.title('Actual vs Predicted Emissions')

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()

# Read the CSV files

def figure_two():
    file1 = pd.read_csv('/Users/yme/code/AppliedAI/summativeassessment/data/lstm_nofs_results.csv')
    file2 = pd.read_csv('/Users/yme/code/AppliedAI/summativeassessment/data/xg_nofs_results.csv')
    file3 = pd.read_csv('/Users/yme/code/AppliedAI/summativeassessment/data/lstm_fs_results.csv')
    file4 = pd.read_csv('/Users/yme/code/AppliedAI/summativeassessment/data/xg_fs_results.csv')

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()

    # Plotting actual and predicted emissions for each file
    for i, ax in enumerate(axs):
        # Select the corresponding file
        if i == 0:
            data = file1
            file_label = 'lstm nofs'
        elif i == 1:
            data = file2
            file_label = 'xg nofs'
        elif i == 2:
            data = file3
            file_label = 'lstm fs'
        elif i == 3:
            data = file4
            file_label = 'xg fs'

        # Plot actual and predicted emissions
        ax.plot(data['Year'], data['Total'], label='Actual')
        ax.plot(data['Year'], data['Predicted Emissions'], label='Predicted')
        ax.set_title(file_label)

        # Add labels and legend only to the last row of subplots
        if i >= len(axs) - 2:
            ax.set_xlabel('Year')
            ax.set_ylabel('Emissions')
            ax.legend()

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()

