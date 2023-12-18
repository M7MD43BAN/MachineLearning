import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

dataframe = pd.read_csv('titanic.csv')

# Drop columns that are not useful for prediction
dataframe = dataframe.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis='columns')
print('------------------Data after drop unuseful columns------------------\n', dataframe.head(n=10), '\n')
print('------------------Data types of the data------------------\n', dataframe.dtypes, '\n')

# Deal with missing and nan values using fillna and mean
mask = dataframe.head(n=10).isna().any(axis=1)
print('------------------Check any nan values------------------\n', mask, '\n')
dataframe.Age = dataframe.Age.fillna(dataframe.Age.mean())
print('------------------Data after filling missing data with mean------------------\n', dataframe.head(n=10), '\n')

# Encoding the categorical data of type binary
dataframe = dataframe.replace({
    'Sex': {'male': 0, 'female': 1},
})
print('------------------Data after numeric encoding------------------\n', dataframe.head(n=10), '\n')

# Splitting the dataset into input features (independent) and output features (dependent)
X = dataframe.iloc[:, :-1].values
y = dataframe.iloc[:, -1].values

# Splitting the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Training the Naive Bayes model on the training set
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the test set results
y_predict = classifier.predict(X_test)

# Concatenate the y_train and y_test
y_combined = np.concatenate((y_predict.reshape(len(y_predict), 1), y_test.reshape(len(y_test), 1)), axis=1)
print('------------------Predicting the test set results------------------\n', y_combined[0:10], '\n')

# Print the accuracy of the model
print('------------------Model Evaluation------------------')
print(f'Accuracy: {classifier.score(X_test, y_test) * 100:.2f}', '%')
