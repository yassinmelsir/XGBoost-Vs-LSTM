import matplotlib.pyplot as plt

results = [
    [1, 0.5, [0.3, 0.4, 0.6], [1, 2, 3], [0.2, 0.3, 0.5]],
    [2, 0.7, [0.8, 0.9, 1.0], [4, 5, 6], [0.7, 0.9, 1.2]],
    # ... more subarrays
]

# Extract RMSE values and run numbers
run_numbers = [result[0] for result in results]
rmse_values = [result[1] for result in results]

# Extract predictions and actual results
predictions = [result[2] for result in results]
actual_outputs = [result[4] for result in results]

# Plot RMSE for each run
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(run_numbers, rmse_values, marker='o')
plt.xlabel('Run Number')
plt.ylabel('RMSE')
plt.title('RMSE for Each Run')

# Plot predictions and actual results
plt.subplot(1, 2, 2)
for i in range(len(results)):
    plt.plot(predictions[i], label=f'Run {run_numbers[i]} Predictions')
    plt.plot(actual_outputs[i], label=f'Run {run_numbers[i]} Actual Outputs')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.title('Predictions and Actual Results')

# Adjust the spacing between subplots
plt.tight_layout()

# Show the figure
plt.show()

#Predictions on dataset with no feature selection

#Predictions with extended dataset

#Predictions on dataset with feature selection

#Predictions runs to dos 
# 10 runs each 
# create three plots: predictions/actuals, runs/rmse, feature_selection results
# plot xg and lstm on same graph
# plot runs with and without feature slection on same graph

# run model 10 times and record results
# results = [full_run_lstm() for _ in range(0,1)]

# #extract run/rmse and predicted/actual/year
# runs, rmses, predicted_emissions, predicted_year, actual_emissions, actual_year = [],[],[],[],[],[]

# data = []
# for i,result in enumerate(results):
#     runs, rmses, predicted_emissions, predicted_year, actual_emissions, actual_year = i,result[0],result[1]['Total'],result[1]['Year'],result[2]['Total'],result[2]['Year']
#     data.append([runs, rmses, predicted_emissions, predicted_year, actual_emissions, actual_year])

# plt.figure(figsize=(8, 4))
# plt.subplot(1, 2, 1)
# plt.plot(runs, rmses)
# plt.xlabel('Run')
# plt.ylabel('RMSE')
# plt.title('RMSE per Run')

# plt.subplot(1, 2, 2)
# for i in range(len(results)):
#     plt.plot(predicted_year,predicted_emissions[i], label=f'Run {runs[i]} Predictions')
#     plt.plot(actual_year,actual_emissions, label=f'Run {runs[i]} Actual Outputs')

# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.legend()
# plt.title('Predictions and Actual Results')

# # Adjust the spacing between subplots
# plt.tight_layout()

# # Show the figure
# plt.show()