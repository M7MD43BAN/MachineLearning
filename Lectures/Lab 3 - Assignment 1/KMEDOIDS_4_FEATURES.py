import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn_extra.cluster import KMedoids

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

# Apply KMedoids clustering algorithm
k_optimal = 5
kmedoids = KMedoids(n_clusters=k_optimal)
kmedoids.fit(dataframe[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
y_predict = kmedoids.labels_
dataframe['Cluster'] = y_predict

# Plot the data points in a scatter plot using matplotlib
plt.figure()
cluster_colors = ['forestgreen', 'crimson', 'royalblue', 'black', 'orange']
for cluster_num in range(k_optimal):
    cluster_data = dataframe[dataframe['Cluster'] == cluster_num]
    plt.scatter(cluster_data['Annual Income (k$)'], cluster_data['Spending Score (1-100)'],
                color=cluster_colors[cluster_num], label=f'Cluster {cluster_num + 1}')

plt.scatter(kmedoids.cluster_centers_[:, 2], kmedoids.cluster_centers_[:, 3], s=150, color='indigo', marker='*',
            label='Medoids')

plt.title('KMedoids Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
