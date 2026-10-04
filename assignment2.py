# Assignment 2
# Clustering using Unsupervised Learning

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
iris = load_iris()

X = iris.data

# Convert to DataFrame
df = pd.DataFrame(
    X,
    columns=iris.feature_names
)

print("Dataset:")
print(df.head())

# Standardize the data
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Create KMeans model
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# Train model
clusters = kmeans.fit_predict(X_scaled)

# Add cluster information
df["Cluster"] = clusters

print("\nClustered Data:")
print(df.head(10))

# Print cluster centers
print("\nCluster Centers:")
print(kmeans.cluster_centers_)

# Visualization
plt.figure(figsize=(8, 6))

plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=clusters,
    cmap="viridis",
    s=50
)

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    color="red",
    marker="X",
    s=200,
    label="Centroids"
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means Clustering")
plt.legend()

plt.show()

# Inertia
print("\nInertia:", kmeans.inertia_)