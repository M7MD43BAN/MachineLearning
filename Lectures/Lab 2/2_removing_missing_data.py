import pandas as pd

data = pd.read_csv(r"dataset/data-example4.csv")

mask = data.isnull().any(axis=1)
# sum function returns the number of true values
number_of_true_values = sum(mask)
print('Number of missing data is: ', number_of_true_values)

# CLean the data by removing the rows with missing values
data_clean = data[~mask]
print('Data after removing missing data \n', data_clean)
