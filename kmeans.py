import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Define the data points
data = np.array([
    [185, 72],  # initial centroid
    [170, 56],  # initial centroid
    [168, 60],
    [179, 68],
    [182, 72],
    [188, 77]
])

# Set initial centroids
initial_centroids = data[:2]  # First two points as initial centroids

# K-Means with 2 clusters, first two centroids as initial centroids, max 2 iterations
kmeans = KMeans(n_clusters=2, init=initial_centroids, n_init=1, max_iter=2)

# Fit the K-Means model
kmeans.fit(data)

# Get the centroids and labels
centroids = kmeans.cluster_centers_  # New centroids
labels = kmeans.labels_  # Cluster labels for each data point

# Plot the data points and centroids
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', label='Data points')  # Plot data points with cluster labels
plt.scatter(centroids[:, 0], centroids[:, 1], s=100, c='red', label='Centroids')  # Plot centroids in red
plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('K-Means Clustering with 2 Clusters')
plt.legend()
plt.show()
