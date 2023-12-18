import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('HousePrices1.csv')

X = data['House Area'].values.reshape(-1, 1)
Y = data['price'].values

# Split the data into 70% training, 15% evaluation, and 15% test sets
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
X_eval, X_test, Y_eval, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

# Use MinMaxScaler to scale data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train).flatten()
X_eval_scaled = scaler.transform(X_eval).flatten()
X_test_scaled = scaler.transform(X_test).flatten()

def batch_gradient_descent(X, Y, learning_rate, epochs):
    m = 0  # slope
    c = 0  # y-intercept

    # Number of data points
    n = float(len(X))

    # Gradient descent for given number of epochs
    for epoch in range(epochs):
        # Calculate predicted values
        Y_predicted = m * X + c

        # Calculate the gradients
        dm = (-2 / n) * np.sum(X * (Y - Y_predicted))
        dc = (-2 / n) * np.sum(Y - Y_predicted)

        # Update parameters using the gradients and learning rate
        m = m - learning_rate * dm
        c = c - learning_rate * dc
    return m, c

# Calculate RMSE
def calculate_rmse(X, y, m, b):
    y_pred = m * X + b
    rmse = np.sqrt(np.mean((y - y_pred)**2))
    return rmse

learning_rate = 0.1
epochs = 1000
slope, intercept = batch_gradient_descent(X_train_scaled, Y_train, learning_rate, epochs)

# Evaluate evaluation set
rmse_eval = calculate_rmse(X_eval_scaled, Y_eval, slope, intercept)
print(f'RMSE on Evaluation Set: {rmse_eval}')

# Evaluate test set
rmse_test = calculate_rmse(X_test_scaled, Y_test, slope, intercept)
print(f'RMSE on Test Set: {rmse_test}')