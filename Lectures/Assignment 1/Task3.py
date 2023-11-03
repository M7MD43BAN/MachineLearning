import pandas as pd
from kmedoids import KMedoids
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Importing the dataset
dataset = pd.read_csv('dish.csv')

# Take 1000 of the data (randomly)
sampled_data = dataset.sample(n=20000, random_state=42)

# Select the features for clustering
features = sampled_data[['first_appeared', 'last_appeared', 'lowest_price', 'highest_price']]

# Taking care of missing data
imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features)

# Standardize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_imputed)

# Perform K-Medoids clustering
k = 25  # Number of clusters
kmedoids = KMedoids(n_clusters=k, random_state=0, metric='euclidean')
kmedoids.fit(scaled_features)

# Add cluster labels to the original dataset
sampled_data['cluster'] = kmedoids.labels_

# Print the resulting clusters
for cluster in range(k):
    print(f"Cluster {cluster}:\n")
    cluster_data = sampled_data[sampled_data['cluster'] == cluster]
    print(cluster_data[['id', 'name']])
    print("\n===========================================\n")

# You can save the results to a new CSV file if needed
sampled_data.to_csv('clustered_sampled_menu_dataset.csv', index=False)
