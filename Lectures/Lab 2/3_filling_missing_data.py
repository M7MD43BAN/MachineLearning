import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

data = pd.read_csv(r"dataset/data-example5.csv")

mask = data.isnull().any(axis=0)
print(mask.sum())
print(data.columns[mask])

_mask = data.isnull().any(axis=1)
print(_mask.sum())
print(data[_mask])

print('The percentage of missing data in the original data: ', _mask.sum() / len(data) * 100, '%')

# Fill the missing data with 0.0
data_2 = data.fillna(0.0)
print(data_2.head())
print("#####################################")

# Fill the missing data with the mean value using SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
data_clean = imputer.fit_transform(data)
print('Data after filling missing data with mean value: \n', data_clean.flatten())
print("#####################################")

# Fill the missing data with the median value using mean function and fillna
data_clean = data.fillna(data.mean(axis=0))
print('Data after filling missing data with mean value: \n', data_clean)
print("#####################################")
