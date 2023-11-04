import numpy as np
from kmedoids import KMedoids

data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)

km = KMedoids(n_clusters=3, metric='manhattan', init='random', random_state=0)
km.fit(data)

cluster_assignments = km.labels_
medoids = km.cluster_centers_.flatten()

for cluster_id in range(3):
    cluster_data = data[cluster_assignments == cluster_id].flatten()
    print(f'Cluster {cluster_id + 1}: {cluster_data}')
    print(f'Medoid of Cluster {cluster_id + 1}: {medoids[cluster_id]}')

