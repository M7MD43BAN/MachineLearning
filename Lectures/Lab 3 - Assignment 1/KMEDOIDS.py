import matplotlib.pyplot as plt
import pandas as pd
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import MinMaxScaler

# Read data from income.csv file using pandas
dataframe = pd.read_csv('Mall_Customers.csv')

# Drop the CustomerID column
dataframe = dataframe.drop(['CustomerID', 'Genre', 'Age'], axis=1)

# Feature scaling using MinMaxScaler (Rescaling)
scaler = MinMaxScaler()
dataframe['Annual Income (k$)'] = scaler.fit_transform(dataframe[['Annual Income (k$)']])
dataframe['Spending Score (1-100)'] = scaler.fit_transform(dataframe[['Spending Score (1-100)']])

# Apply KMeans clustering algorithm
k_optimal = 5
kmedoids = KMedoids(n_clusters=k_optimal)
kmedoids.fit(dataframe[['Annual Income (k$)', 'Spending Score (1-100)']])
y_predict = kmedoids.labels_

dataframe['Cluster'] = y_predict

# Plot the data points in a scatter plot using matplotlib
plt.figure()
cluster_colors = ['forestgreen', 'crimson', 'royalblue', 'black', 'orange']
for cluster_num in range(k_optimal):
    cluster_data = dataframe[dataframe['Cluster'] == cluster_num]
    plt.scatter(cluster_data['Annual Income (k$)'], cluster_data['Spending Score (1-100)'],
                color=cluster_colors[cluster_num], label=f'Cluster {cluster_num + 1}')

plt.scatter(kmedoids.cluster_centers_[:, 0], kmedoids.cluster_centers_[:, 1], color='indigo', marker='*', label='Medoids')

plt.title('KMedoids Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
