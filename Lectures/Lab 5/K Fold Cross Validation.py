import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

dataframe = pd.read_csv('Social_Network_Ads.csv')

# Splitting the dataset into input features (independent) and output features (dependent)
X = dataframe.iloc[:, :-1].values
y = dataframe.iloc[:, -1].values

# Splitting the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature scaling (not necessary for Naive Bayes, but good practice)
# As Naive Bayes algorithm is based on probability not on distance, so it doesn't require feature scaling.
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Naive Bayes model on the training set
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting a new result
print(classifier.predict(sc.transform([[30, 87000]])))

# Predicting the test set results
y_predict = classifier.predict(X_test)
print(np.concatenate((y_predict.reshape(len(y_predict), 1), y_test.reshape(len(y_test), 1)), 1))

print(classifier.score(X_test, y_test) * 100, '%')

# Making the confusion matrix
y_predict = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict)
print(cm)

print(accuracy_score(y_test, y_predict) * 100, '%')

# Applying K-Fold Cross Validation
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print("Accuracy: {:.2f} %".format(accuracies.mean() * 100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()))
