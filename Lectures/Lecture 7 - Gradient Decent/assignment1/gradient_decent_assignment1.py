import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('HousePrices1.csv')
features = data["House Area"].values.reshape(-1, 1)
target = data["price"].values

scaler = StandardScaler()
features = scaler.fit_transform(features)

X_train, X_temp, y_train, y_temp = train_test_split(features, target, test_size=0.4, random_state=42)
X_eval, X_test, y_eval, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

weights = np.zeros(X_train.shape[1])
bias = 0.0

learning_rate = 0.01
epochs = 1000


def batch_gradient_descent(X, y, weights, bias, learning_rate):
    m = len(X)
    dw = np.zeros_like(weights)
    db = 0.0

    for xi, yi in zip(X, y):
        dw += -2 * xi * (yi - np.dot(weights, xi) - bias)
        db += -2 * (yi - np.dot(weights, xi) - bias)

    weights -= learning_rate * (1 / m) * dw
    bias -= learning_rate * (1 / m) * db

    return weights, bias


for epoch in range(epochs):
    weights, bias = batch_gradient_descent(X_train, y_train, weights, bias, learning_rate)

eval_predictions = np.dot(X_eval, weights) + bias

rmse_eval = np.sqrt(mean_squared_error(y_eval, eval_predictions))
print(f'RMSE on the evaluation set: {rmse_eval}')

test_predictions = np.dot(X_test, weights) + bias

rmse_test = np.sqrt(mean_squared_error(y_test, test_predictions))
print(f'RMSE on the test set: {rmse_test}')

print('Actual values: \n', y_test[:10])
print('Predicted values: \n', np.round(test_predictions[:10], 3))
