import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# Read data from income.csv file using pandas
dataframe = pd.read_csv('Mall_Customers.csv')

# Drop the CustomerID column
dataframe = dataframe.drop(['CustomerID'], axis=1)

# Rename the Genre column
dataframe = dataframe.rename(columns={'Genre': 'Gender'})

# Encoding the gender (Binary data)
dataframe = dataframe.replace({
    'Gender': {'Male': 0, 'Female': 1}
})

# Determine the number of clusters using elbow method
WCSS = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=1)
    kmeans.fit(dataframe[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
    WCSS.append(kmeans.inertia_)

plt.figure()
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_range, WCSS)

# Apply KMeans clustering algorithm
k_optimal = 5
kmeans = KMeans(n_clusters=k_optimal)
y_predict = kmeans.fit_predict(dataframe[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
dataframe['Cluster'] = y_predict

# Plot the data points in a scatter plot using matplotlib
plt.figure()
cluster_colors = ['forestgreen', 'crimson', 'royalblue', 'black', 'orange']
for cluster_num in range(k_optimal):
    cluster_data = dataframe[dataframe['Cluster'] == cluster_num]
    plt.scatter(cluster_data['Annual Income (k$)'], cluster_data['Spending Score (1-100)'],
                color=cluster_colors[cluster_num], label=f'Cluster {cluster_num + 1}')

plt.scatter(kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:, 3], color='indigo', marker='*', label='Centroids')

plt.title('KMeans Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
