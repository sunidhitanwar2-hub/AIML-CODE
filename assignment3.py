# Assignment 3
# Dimensionality Reduction using PCA

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load dataset
iris = load_iris()

X = iris.data
y = iris.target

print("Original shape of dataset:")
print(X.shape)

# Standardize data
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("\nShape after PCA:")
print(X_pca.shape)

# Create DataFrame
pca_df = pd.DataFrame(
    X_pca,
    columns=["Principal Component 1", "Principal Component 2"]
)

pca_df["Target"] = y

print("\nPCA Data:")
print(pca_df.head())

# Explained variance
print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

print("\nTotal Explained Variance:")
print(sum(pca.explained_variance_ratio_))

# Visualization
plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y,
    cmap="viridis",
    s=50
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title("PCA Dimensionality Reduction")

plt.colorbar(label="Class")

plt.grid(True)

plt.show()