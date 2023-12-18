import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Read data
data = pd.read_csv('HousePrices2.csv')

# Feature matrix and target variable
x = data[['House Area', 'Rooms', 'floor']].values
y = data['price'].values


# Splitting data into train, validation, and test sets
x_train, x_remain_data, y_train, y_remain_data = train_test_split(x, y, train_size=0.6, test_size=0.4, random_state=1)
x_vali, x_test, y_vali, y_test = train_test_split(x_remain_data, y_remain_data, train_size=0.5, test_size=0.5, random_state=1)

# Scaling data
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_vali = sc.transform(x_vali)
x_test = sc.transform(x_test)

# Adding bias term to x extra column
x_train = np.column_stack((np.ones(len(x_train)), x_train))
x_vali = np.column_stack((np.ones(len(x_vali)), x_vali))
x_test = np.column_stack((np.ones(len(x_test)), x_test))

# calculate cost (RMSE)
def cal_cost(x, y, thetas):
    predictions = np.dot(x, thetas)
    mse_test = np.square(np.subtract(predictions , y)).mean()
    rmse_test = np.sqrt(mse_test)
    return rmse_test


# Gradient Descent on training set
def cal_GD(x, y, thetas):
    learning_rate = 0.1
    epochs = 10000
    n = len(y)
    costs = []  # To store costs for each epoch
    for epoch in range(epochs):
        predictions = np.dot(x, thetas)
        slopes = np.dot(x.T,  predictions - y)* 2/ n
        thetas -= learning_rate * slopes

        # Calculate and store cost for analysis
        cost = cal_cost(x, y, thetas)
        costs.append(cost)

    return thetas, costs

# execute gradient descent on training set
initial_thetas = np.zeros(x_train.shape[1])
thetas_result, _ = cal_GD(x_train, y_train, initial_thetas)

print("theat 0: ", thetas_result[0]," \ntheat 1:", thetas_result[1]," \ntheat 2:", thetas_result[2])

# Applying thetas to validation set
vali_cost = cal_cost(x_vali, y_vali, thetas_result)
print("RMSE on Validation set :", vali_cost)

# Applying thetas to test set
test_cost = cal_cost(x_test, y_test, thetas_result)
print("RMSE on Test set :", test_cost)
