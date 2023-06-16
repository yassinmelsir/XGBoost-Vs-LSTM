import pandas as pd
df = pd.read_csv('/Users/yme/code/AppliedAI/summativeassessment/data/full_dataset.csv').fillna(0).drop('ISO', axis=1)
