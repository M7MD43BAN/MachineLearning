import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def RMSE(X, y, theta_0, theta_1):
    y_predict = theta_0 + theta_1 * X
    mse = (np.square(np.subtract(y, y_predict))).mean()
    rmse = np.sqrt(mse)

    return rmse


def batch_gradient_descent(X, y, L_rate, epochs):
    theta_0 = 0
    theta_1 = 0
    n = len(X)

    for i in range(epochs):
        y_pred = theta_0 + theta_1 * X

        d_theta0 = (2 / n) * np.sum(y_pred - y)
        d_theta1 = (2 / n) * np.sum(X * (y_pred - y))

        theta_0 = theta_0 - L_rate * d_theta0
        theta_1 = theta_1 - L_rate * d_theta1

    return theta_0, theta_1


data = pd.read_csv('HousePrices1.csv')
X = data['House Area'].values.reshape(-1, 1)
y = data['price'].values.reshape(-1, 1)

X_mean = np.mean(X)
y_mean = np.mean(y)
X_standard_deviation = np.std(X)
y_standard_deviation = np.std(y)
X_scaled = ((X - X_mean) / X_standard_deviation).flatten()
y_scaled = ((y - y_mean) / y_standard_deviation).flatten()

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X).flatten()
# y_scaled = scaler.fit_transform(y).flatten()

X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_scaled, test_size=0.4, random_state=42)
X_eval, X_test, y_eval, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

L_rate = 0.003
epochs = 1000

theta_0, theta_1 = batch_gradient_descent(X_train, y_train, L_rate, epochs)

mse_eval = RMSE(X_eval, y_eval, theta_0, theta_1)
print(f'RMSE on the evaluation set: ', mse_eval)

mse_test = RMSE(X_test, y_test, theta_0, theta_1)
print(f'RMSE on the test set: ', mse_test)
