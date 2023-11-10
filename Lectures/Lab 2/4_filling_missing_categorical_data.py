import pandas as pd

data = pd.read_csv(r"dataset/data-example6.csv")

print(data)

most_frequent_value = data['x3'].mode()[0]
print('Most frequent value is: ', most_frequent_value)

# Fill the missing values in the column 'x3' with the most frequent value
data_2 = data.fillna({'x3': most_frequent_value})
print('Data after filling missing values: \n', data_2)
