# import libs
import pandas as pd
from impyute.imputation.cs import mice


# training data
train = pd.read_csv('train.csv')

# test data
test = pd.read_csv('test.csv')

# print(train.describe())
mice(train.values).to_csv('train11.csv')