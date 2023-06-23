# tool test with randomly country with random renewable data 
from functions import gen_tool_dataset
from figures import figure_for_file
from analysis import xg_tool, lstm_tool


# AI TOOL
# generate a dataset for UK emissions/renewable energy, curve the renewable energy columns
# , and observe the predicted CO2 emission impact

xg_features = [2, 3, 4, 5, 8, 9, 11, 12, 13, 14, 15, 17, 19, 20, 25] # xg features
lstm_features = [2, 4, 6, 9, 10, 16, 17, 18, 19, 23, 24, 28] # lstm features
input_file = 'test/UK-XG' # input test file name
output_file = 'UK-XG-Result' # out result file name

#dataset generation
gen_tool_dataset(input_file) # curve data here
# model to use
xg_tool(output_file,input_file,lstm_features) 
#figure generation
figure_for_file(output_file)
