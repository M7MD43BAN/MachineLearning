import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def RMSE(X, y, theta):
    y_predict = np.dot(X, theta)
    mse = np.square(np.subtract(y, y_predict)).mean()
    rmse = np.sqrt(mse)
    return rmse


def batch_gradient_descent(X, y, learning_rate, epochs):
    n = len(X)
    theta = np.zeros(X.shape[1])

    for epoch in range(epochs):
        y_pred = np.dot(X, theta)
        gradient = np.dot(X.T, y_pred - y) * 2 / n
        theta -= learning_rate * gradient

    return theta


data = pd.read_csv('HousePrices2.csv')
X = data[['House Area', 'Rooms', 'floor']].values
y = data['price'].values

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=1)
X_eval, X_test, y_eval, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1)

# Feature scaling using mean normalization
mean = np.mean(X_train, axis=0)
standard_deviation = np.std(X_train, axis=0)

X_train_scaled = (X_train - mean) / standard_deviation
X_eval_scaled = (X_eval - mean) / standard_deviation
X_test_scaled = (X_test - mean) / standard_deviation

# Add a column of ones to X for the bias term
X_train_scaled = np.column_stack((np.ones(len(X_train_scaled)), X_train_scaled))
X_eval_scaled = np.column_stack((np.ones(len(X_eval_scaled)), X_eval_scaled))
X_test_scaled = np.column_stack((np.ones(len(X_test_scaled)), X_test_scaled))

learning_rate = 0.1
epochs = 1000

thetas = batch_gradient_descent(X_train_scaled, y_train, learning_rate, epochs)

# Print thetas for this formula "price = theta0 + theta1 * house area + theta2 * rooms + theta3 * floor"
print("Theta 0: ", thetas[0], "\nTheta 1:", thetas[1], "\nTheta 2:", thetas[2])

# Evaluate on the evaluation set
rmse_eval = RMSE(X_eval_scaled, y_eval, thetas)
print(f'RMSE on the evaluation set: {rmse_eval}')

# Evaluate on the test set
rmse_test = RMSE(X_test_scaled, y_test, thetas)
print(f'RMSE on the test set: {rmse_test}')
