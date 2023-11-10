import pandas as pd

data = pd.read_csv(r"dataset/data-example7.csv")

print('Data before encoding: \n', data.head())
print("-------------------------------------------------------------")

# Data types of the dataset
data_types = data.dtypes
print('Data types: \n', data_types)
print("-------------------------------------------------------------")

# Print the unique values of the column 'x1', 'x2' and 'x4'
print('Unique values of x1: ', data['x1'].unique())
print('Unique values of x3: ', data['x3'].unique())
print('Unique values of x4: ', data['x4'].unique())
print("-------------------------------------------------------------")

# Encoding the categorical data of type nominal
data_encoded = data.replace({
    'x1': {'male': 0, 'female': 1},
    'x3': {'low': 0, 'med': 1, 'high': 2},
})

print('Data after numeric encoding: \n', data_encoded.head())
print("-------------------------------------------------------------")

# Encoding the categorical data of type ordinal with data type int and prefix 'is'
data_encoded_2 = pd.get_dummies(data_encoded, dtype=int, prefix='is')
print('Data after one-hot encoding: \n', data_encoded_2.head())
