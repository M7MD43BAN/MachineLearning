import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

data = np.array([
    [9, 3],
    [8, 4],
    [4, 6],
    [8, 5],
    [2, 5],
    [3, 8],
    [5, 8],
    [4, 4],
    [10, 4],
    [9, 6]
])

# n_clusters=3 specifies that we want to partition the data into 3 clusters.
# init='random' specifies that we want to randomly initialize the cluster centroids.
# random_state=0 to get the same result each time we run the code.
kmeans = KMeans(n_clusters=3, init='random', random_state=0)
kmeans.fit(data)

cluster_assignments = kmeans.labels_
cluster_centroids = kmeans.cluster_centers_

for cluster_id in range(3):
    cluster_data = data[cluster_assignments == cluster_id]
    print(f'Cluster {cluster_id + 1} {cluster_data}')
    print(f'Centroid of Cluster {cluster_id + 1}: {cluster_centroids[cluster_id]}')

plt.figure(figsize=(8, 6))
for cluster_id in range(3):
    cluster_data = data[cluster_assignments == cluster_id]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster_id + 1}')

plt.scatter(cluster_centroids[:, 0], cluster_centroids[:, 1], c='red', marker='*', s=100, label='Centroids')

plt.title('K-Means Clustering')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.legend()
plt.show()
