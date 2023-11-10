import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

# Read the dataset and print it
dataset = pd.read_csv(r"dataset/Data.csv")
print('------------------Dataset------------------\n', dataset, '\n')

# Divide the dataset into input and output
X = dataset.iloc[:, :-1].values  # Get all rows and all columns except the last one - the input
Y = dataset.iloc[:, 3].values  # Get all rows and the last column - the output

X_dataframe = pd.DataFrame(X, columns=['Country', 'Age', 'Salary'])
Y_dataframe = pd.DataFrame(Y, columns=['Purchased'])

print('------------------Input Data------------------\n', X_dataframe, '\n')
print('------------------Output Data------------------\n', Y_dataframe, '\n')

# Deal with missing data using SimpleImputer with mean
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])
X_dataframe = pd.DataFrame(X, columns=['Country', 'Age', 'Salary'])
print('------------------Input Data after dealing with missing data------------------\n', X_dataframe, '\n')

# Encode categorical data of type nominal using one-hot encoding
country_columns = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(country_columns.fit_transform(X))

X_dataframe = pd.DataFrame(X, columns=['France', 'Germany', 'Spain', 'Age', 'Salary'])
print('------------------Input Data after encoding categorical data------------------\n', X_dataframe, '\n')

# Deal with missing data using
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y)

Y_dataframe = pd.DataFrame(Y, columns=['Purchased'])
print('------------------Output Data after encoding categorical data------------------\n', Y_dataframe, '\n')

# Feature scaling
standard_scaler = StandardScaler()
X[:, 3:] = standard_scaler.fit_transform(X[:, 3:])
X_dataframe = pd.DataFrame(X, columns=['France', 'Germany', 'Spain', 'Age', 'Salary'])
print('------------------Input Data after feature scaling------------------\n', X_dataframe, '\n')
