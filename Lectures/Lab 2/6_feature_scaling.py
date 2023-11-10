import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv(r"dataset/data-example8.csv")

data_input = data.drop(columns='y')
data_output = data['y']

X, X_test, y, y_test = train_test_split(data_input, data_output, test_size=0.2, random_state=42)

print('X: ', X.shape)
print('X_test: ', X_test.shape)
print('y: ', y.shape)
print('y_test: ', y_test.shape)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to dataframe
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
print(X_scaled.head())
