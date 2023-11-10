import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn_extra.cluster import KMedoids

# Read data from income.csv file using pandas
dataframe = pd.read_csv('income.csv')

# Plot the data points in a scatter plot using matplotlib
plt.figure()
plt.scatter(dataframe['Age'], dataframe['Income($)'])
plt.xlabel('Age')
plt.ylabel('Income($)')

# Feature scaling using MinMaxScaler (Rescaling) between 0 and 1
scaler = MinMaxScaler()
dataframe['Income($)'] = scaler.fit_transform(dataframe[['Income($)']])
dataframe['Age'] = scaler.fit_transform(dataframe[['Age']])

# Apply KMedoids clustering algorithm
kmedoids = KMedoids(n_clusters=3)
kmedoids.fit(dataframe[['Age', 'Income($)']])
y_predict = kmedoids.labels_

# Add a new column to the dataframe with the cluster number
dataframe['Cluster'] = y_predict

dataframe_1 = dataframe[dataframe['Cluster'] == 0]
dataframe_2 = dataframe[dataframe['Cluster'] == 1]
dataframe_3 = dataframe[dataframe['Cluster'] == 2]

plt.figure()
medoids = dataframe.iloc[kmedoids.medoid_indices_]
plt.scatter(dataframe_1['Age'], dataframe_1['Income($)'], color='red', label='Cluster 1')
plt.scatter(dataframe_2['Age'], dataframe_2['Income($)'], color='green', label='Cluster 2')
plt.scatter(dataframe_3['Age'], dataframe_3['Income($)'], color='blue', label='Cluster 3')
plt.scatter(medoids['Age'], medoids['Income($)'], color='magenta', marker='*', s=120, label='Medoids')

plt.xlabel('Age')
plt.ylabel('Income($)')
plt.legend()
plt.show()
