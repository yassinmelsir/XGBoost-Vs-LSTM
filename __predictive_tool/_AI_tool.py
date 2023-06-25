# tool test with randomly country with random renewable data 
from functions import gen_tool_dataset
from figures import figure_for_file
from analysis import xg_tool, lstm_tool
import numpy as np

# AI TOOL
# generate a dataset for UK emissions/renewable energy, curve the renewable energy columns
# , and observe the predicted CO2 emission impact

# change features to column names
feature_1 = [8, 9, 11, 12, 13, 14, 15, 17, 19, 20, 25] # xg features
# feature_1 = np.arange(8,25,1)
input_file = 'test/UK-normal' # input test file name 
output_file = 'UK-normal-xg' # out result file name

# Implement data manipulation for prediction here

#dataset generation
gen_tool_dataset(input_file) # curve data here
# model to use
xg_tool(output_file,input_file,feature_1)
lstm_tool(output_file,input_file,feature_1)

#figure generation
figure_for_file(output_file)
