import pandas as pd
import matplotlib.pyplot as plt

def multi_figure(file_1_name,file_2_name,file_3_name,file_4_name):
    file1 = pd.read_csv(f'/Users/yme/code/AppliedAI/summativeassessment/data/results/{file_1_name}.csv')
    file2 = pd.read_csv(f'/Users/yme/code/AppliedAI/summativeassessment/data/results/{file_2_name}.csv')
    file3 = pd.read_csv(f'/Users/yme/code/AppliedAI/summativeassessment/data/results/{file_3_name}.csv')
    file4 = pd.read_csv(f'/Users/yme/code/AppliedAI/summativeassessment/data/results/{file_4_name}.csv')

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()

    for i, ax in enumerate(axs):
        if i == 0:
            data = file1
            file_label = file_1_name
        elif i == 1:
            data = file2
            file_label = file_2_name
        elif i == 2:
            data = file3
            file_label = file_3_name
        elif i == 3:
            data = file4
            file_label = file_4_name

        ax.plot(data['Year'], data['Total'], label='Actual')
        ax.plot(data['Year'], data['Predicted Emissions'], label='Predicted')
        ax.set_title(file_label)

        if i >= len(axs) - 2:
            ax.set_xlabel('Year')
            ax.set_ylabel('Emissions')
            ax.legend()

    plt.tight_layout()
    plt.show()

def figure_for_file(filename):
    file = pd.read_csv(f'/Users/yme/code/AppliedAI/summativeassessment/data/results/{filename}.csv')

    plt.figure(figsize=(10, 6))
    plt.plot(file['Year'], file['Total'], label='Actual')
    plt.plot(file['Year'], file['Predicted Emissions'], label='Predicted')
    plt.xlabel('Year')
    plt.ylabel('Emissions')
    plt.title('Actual vs Predicted Emissions')
    plt.legend()
    plt.show()

