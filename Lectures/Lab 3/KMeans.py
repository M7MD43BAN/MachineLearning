from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

dataframe = pd.read_csv('income.csv')

plt.figure()

plt.scatter(dataframe['Age'], dataframe['Income($)'])
plt.xlabel('Age')
plt.ylabel('Income($)')

kmeans = KMeans(n_clusters=4)
y_predict_1 = kmeans.fit_predict(dataframe[['Age', 'Income($)']])

dataframe['Cluster'] = y_predict_1
print(dataframe)

print(kmeans.cluster_centers_)

dataframe_1 = dataframe[dataframe['Cluster'] == 0]
dataframe_2 = dataframe[dataframe['Cluster'] == 1]
dataframe_3 = dataframe[dataframe['Cluster'] == 2]
dataframe_4 = dataframe[dataframe['Cluster'] == 3]

plt.figure()

plt.scatter(dataframe_1['Age'], dataframe_1['Income($)'], color='green', label='Cluster 1')
plt.scatter(dataframe_2['Age'], dataframe_2['Income($)'], color='red', label='Cluster 2')
plt.scatter(dataframe_3['Age'], dataframe_3['Income($)'], color='black', label='Cluster 3')
plt.scatter(dataframe_4['Age'], dataframe_4['Income($)'], color='blue', label='Cluster 4')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='purple', marker='*', label='Centroids')

plt.xlabel('Age')
plt.ylabel('Income($)')
plt.legend()

# Feature scaling
scaler = MinMaxScaler()
dataframe['Income($)'] = scaler.fit_transform(dataframe[['Income($)']])
dataframe['Age'] = scaler.fit_transform(dataframe[['Age']])

kmeans = KMeans(n_clusters=4)
y_predict_2 = kmeans.fit_predict(dataframe[['Age', 'Income($)']])

dataframe['Cluster'] = y_predict_2
print(dataframe)

print(kmeans.cluster_centers_)

dataframe_1 = dataframe[dataframe['Cluster'] == 0]
dataframe_2 = dataframe[dataframe['Cluster'] == 1]
dataframe_3 = dataframe[dataframe['Cluster'] == 2]
dataframe_4 = dataframe[dataframe['Cluster'] == 3]

plt.figure()

plt.scatter(dataframe_1['Age'], dataframe_1['Income($)'], color='green', label='Cluster 1')
plt.scatter(dataframe_2['Age'], dataframe_2['Income($)'], color='red', label='Cluster 2')
plt.scatter(dataframe_3['Age'], dataframe_3['Income($)'], color='black', label='Cluster 3')
plt.scatter(dataframe_4['Age'], dataframe_4['Income($)'], color='blue', label='Cluster 4')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='purple', marker='*', label='Centroids')

plt.xlabel('Age')
plt.ylabel('Income($)')
plt.legend()

WCSS = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(dataframe[['Age', 'Income($)']])
    WCSS.append(kmeans.inertia_)

plt.figure()
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_range, WCSS)

kmeans = KMeans(n_clusters=3)
y_predict_3 = kmeans.fit_predict(dataframe[['Age', 'Income($)']])

dataframe['Cluster'] = y_predict_3
print(dataframe)

print(kmeans.cluster_centers_)

dataframe_1 = dataframe[dataframe['Cluster'] == 0]
dataframe_2 = dataframe[dataframe['Cluster'] == 1]
dataframe_3 = dataframe[dataframe['Cluster'] == 2]

plt.figure()

plt.scatter(dataframe_1['Age'], dataframe_1['Income($)'], color='green', label='Cluster 1')
plt.scatter(dataframe_2['Age'], dataframe_2['Income($)'], color='red', label='Cluster 2')
plt.scatter(dataframe_3['Age'], dataframe_3['Income($)'], color='black', label='Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='purple', marker='*', label='Centroids')

plt.xlabel('Age')
plt.ylabel('Income($)')
plt.legend()
plt.show()
