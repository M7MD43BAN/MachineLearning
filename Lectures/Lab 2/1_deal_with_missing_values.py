import pandas as pd

data = pd.read_csv(r"dataset/data-example.csv", na_values=["?", "UNDEFINED"])

# Print the first 5 rows of the data
data_head = data.head
print(data_head)
print("#####################################")

# Print the data types of the columns
data_types = data.dtypes
print(data_types)
print("#####################################")

# Print true if the data is missing value, false if not
mask = data.isnull()
print(mask)
print("#####################################")

# any function returns true if any value is missing value in the row or column
# axis=0 means column, axis=1 means row
print(data[mask.any(axis=1)])
print("#####################################")

# ~ is the not operator to return the opposite of the mask
print(data[~mask.any(axis=1)])
print("#####################################")

# Print the columns and its data that have missing values
_mask = data.isnull().any(axis=0)
columns_with_missing_values = data.columns[_mask]
print(columns_with_missing_values)
print(data[columns_with_missing_values])
print("#####################################")

