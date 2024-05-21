import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from umap import UMAP
from scipy.spatial import ConvexHull


def umap_test():
    # Load sample dataset (Iris dataset)
    data = load_iris()
    X = data.data
    y = data.target

    # Apply UMAP
    umap = UMAP(n_neighbors=5, min_dist=0.3, n_components=2)
    X_umap = umap.fit_transform(X)

    # Plot the results
    plt.figure(figsize=(8, 6))
    for i in range(len(np.unique(y))):
        plt.scatter(X_umap[y == i, 0], X_umap[y == i, 1], label=data.target_names[i])
    plt.title('UMAP Visualization of Iris Dataset')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.legend()
    plt.show()


def convex_hull(cluster_points):
    # Example 2D cluster points (replace with your actual data)
    cluster_points = np.array([[1, 2], [2, 3], [2, 2], [8, 7], [7, 8]])

    # Compute convex hull of cluster points
    hull = ConvexHull(cluster_points)

    # Plot the cluster points
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=10, color='blue', label='Cluster Points')

    # Plot the convex hull
    for simplex in hull.simplices:
        plt.plot(cluster_points[simplex, 0], cluster_points[simplex, 1], 'r--', linewidth=2)

    # Add labels and legend
    plt.title('Convex Hull around Cluster Points')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()

    # Show plot
    plt.show()


if __name__ == "__main__":
    convex_hull("")
