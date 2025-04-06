import random
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans, DBSCAN
from scipy.spatial import Delaunay
from shapely.geometry import Polygon, MultiPoint
from shapely.ops import cascaded_union, polygonize
from collections import defaultdict
import math


def sparsity_analysis_kde(x, y, threshold, n_clusters):
    # Perform kernel density estimation
    kde = gaussian_kde([x, y])

    # Define grid points for density estimation
    x_grid = np.linspace(min(x), max(x), 100)
    y_grid = np.linspace(min(y), max(y), 100)
    x_grid, y_grid = np.meshgrid(x_grid, y_grid)
    density_grid = kde(np.vstack([x_grid.ravel(), y_grid.ravel()]))

    # Reshape density grid for plotting
    density_grid = density_grid.reshape(x_grid.shape)
    # threshold = 0.05  # Adjust as needed
    while True:
        high_density_points = np.argwhere(density_grid > threshold)
        if len(high_density_points) > 0:
            break
        else:
            threshold /= 10

    # Extract coordinates of high density points
    coordinates = [[x_grid[idx[0], idx[1]], y_grid[idx[0], idx[1]]] for idx in high_density_points]

    # Perform KMeans clustering
    n_clusters = n_clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(coordinates)

    # Calculate distance among clusters
    cluster_centers = kmeans.cluster_centers_
    distances = euclidean_distances(cluster_centers)
    distances_ = []
    for i in range(len(distances)):
        dist = []
        for j in range(len(distances[i])):
            if i != j:
                dist.append(distances[i][j])
        distances_.append(dist)

    print("Cluster centers: ", cluster_centers)
    print("Distance cal:", distances_)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    sns.heatmap(density_grid, cmap='jet', alpha=0.8, ax=axs[0])
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')
    axs[0].invert_yaxis()
    axs[0].set_title('Kernel Density Estimation')
    # Plot pairwise distances
    for z in range(len(distances_)):
        axs[1].plot(range(len(distances_[z])), distances_[z], marker='o',
                    linestyle='-')  # [random.randint(0, len(distances) -1)]
    # axs[1].plot(range(len(distances_)), distances_, marker='o', linestyle='-')
    axs[1].set_title('Cluster Distances')
    axs[1].set_xlabel('Point Index')
    axs[1].set_ylabel('Distance')
    axs[1].set_ylim(0)
    coordinates_ = np.asarray(coordinates)
    for i in range(n_clusters):
        axs[2].scatter(coordinates_[labels == i, 0], coordinates_[labels == i, 1], label=f'Cluster {i + 1}')
    axs[2].scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='X', s=100, c='black', label='Centroids')
    axs[2].set_title('KMeans Clusters')
    axs[2].set_xlabel('Point Index')
    axs[2].set_ylabel('Point Index')

    # Show plot
    plt.tight_layout()
    plt.show()


# Function to calculate alpha shape (concave hull)
def alpha_shape(points, alpha):
    if len(points) < 4:
        # If fewer than 4 points, directly return 0 area or treat as a degenerate case
        return 0
    try:
        # Perform Delaunay triangulation
        tri = Delaunay(points)  # qhull_options='QJ')
        # Find edges of the triangles
        edges = defaultdict(float)
        for ia, ib, ic in tri.simplices:
            a = points[ia]
            b = points[ib]
            c = points[ic]
            # Lengths of sides of the triangle
            ab = np.linalg.norm(a - b)
            bc = np.linalg.norm(b - c)
            ca = np.linalg.norm(c - a)

            # Calculate the circumradius (radius of the circle circumscribing the triangle)
            s = (ab + bc + ca) / 2.0
            area = math.sqrt(s * (s - ab) * (s - bc) * (s - ca))
            if area == 0:
                circum_r = np.inf
            else:
                circum_r = (ab * bc * ca) / (4.0 * area)
            # Keep the edge if the circumradius is smaller than 1/alpha
            if circum_r < 1.0 / alpha:
                edges[tuple(sorted([ia, ib]))] += 1
                edges[tuple(sorted([ib, ic]))] += 1
                edges[tuple(sorted([ic, ia]))] += 1

        # Create a set of boundary edges
        boundary_edges = [key for key, val in edges.items() if val == 1]

        # Get the boundary points from the edges
        boundary_points = []
        for ia, ib in boundary_edges:
            boundary_points.append(points[ia])
            boundary_points.append(points[ib])

        # If there are fewer than 3 boundary points, return area = 0
        if len(boundary_points) < 3:
            return 0

        # Use Shapely to calculate the area of the polygon formed by the boundary points
        boundary_polygon = MultiPoint(boundary_points).convex_hull

        return boundary_polygon
    except Exception as e:
        return 0


def sparsity_analysis_from_raw_data(pca_result, alg_type, plt_title="", number_of_clusters=4):
    labels = []
    if alg_type == "kmeans":
        kmeans = KMeans(n_clusters=number_of_clusters, random_state=55)
        labels = kmeans.fit_predict(pca_result)
        cluster_points = []
        print("Labels: ", labels)
        # Calculate distance among clusters
        cluster_centers = kmeans.cluster_centers_
    else:
        n_clusters_ = 0
        eps_init = 0.1
        while n_clusters_ == 0:
            dbscan = DBSCAN(eps=eps_init, min_samples=4)
            dbscan.fit(pca_result)

            labels = dbscan.labels_
            core_samples_mask = np.zeros_like(labels, dtype=bool)
            core_samples_mask[dbscan.core_sample_indices_] = True

            # Number of clusters in labels, ignoring noise if present.
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            eps_init += 0.01
            print("Checking for new clusters ...")
        # Get cluster centers (centroids) by computing the mean of each cluster

        cluster_centers = []
        cluster_points = []
        if n_clusters_ > 0:
            # cluster_points = [pca_result[labels == jj] for jj in range(n_clusters_)]
            for z in range(n_clusters_):
                cluster_points_ = pca_result[labels == z]  # Points belonging to the cluster
                cluster_center = np.mean(cluster_points_, axis=0)  # Compute centroid of cluster
                cluster_centers.append(cluster_center)
                cluster_points.append(cluster_points_)
        cluster_centers = np.asarray(cluster_centers)
        number_of_clusters = n_clusters_
    print("Number of Clusters: ", number_of_clusters)
    # print(f"Cluster centers: {cluster_centers}")
    distances = euclidean_distances(cluster_centers)
    # print(f"Cluster distances: {distances}")
    distances_ = []
    for i in range(len(distances)):
        dist = []
        for j in range(len(distances[i])):
            if i != j:
                dist.append(distances[i][j])
        distances_.append(dist)

    # print("Cluster centers: ", cluster_centers)
    print("Distance to other clusters (excluding the cluster itself):", distances_)
    avg_clu_dist_all = []
    for zx in range(len(distances_)):
        avg_distance_in_cluster = sum(distances_[zx])/len(distances_[zx])
        # print(f"Avg distance between a cluster and its and other clusters: {avg_distance_in_cluster}")
        avg_clu_dist_all.append(avg_distance_in_cluster)
    print(f"Average distance between clusters: {sum(avg_clu_dist_all)/len(avg_clu_dist_all)}")
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    colors = np.random.rand(number_of_clusters, 3)
    if pca_result.shape[1] > 2:
        axs[0] = fig.add_subplot(121, projection='3d')
    for i in range(number_of_clusters):
        boundary_polygon = alpha_shape(cluster_points[i], 0.5)
        if boundary_polygon != 0:
            print(f"Area (Alpha shape): {boundary_polygon.area}")
        if pca_result.shape[1] > 2:
            axs[0].scatter(pca_result[labels == i, 0], pca_result[labels == i, 1], pca_result[labels == i, 2],
                           c=colors[i],
                           label=f'Cluster {i + 1}')
            axs[0].scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2], marker='X', s=10,
                           c='black', label='Centroids')

        else:
            axs[0].scatter(pca_result[labels == i, 0], pca_result[labels == i, 1], c=colors[i],
                           label=f'Cluster {i + 1}')
            axs[0].scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='X', s=10, c='black', label='Centroids')
        if boundary_polygon != 0:
            x, y = boundary_polygon.exterior.xy
            axs[0].plot(x, y, 'r-', label=f'Alpha Shape (alpha={0.5})')

        # ############################################################################
        clust_dist = euclidean_distances(cluster_points[i])
        clust_dist_ = []
        for i in range(len(clust_dist)):
            dist = []
            for j in range(len(clust_dist[i])):
                if i != j:
                    dist.append(clust_dist[i][j])
            clust_dist_.extend(dist)
        # print(distances_)
        avg_distance_clust_pts = sum(clust_dist_)/len(clust_dist_)
        print(f"Clus pts avg dis: {avg_distance_clust_pts}")
        # ############################################################################

    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')
    if pca_result.shape[1] > 2:
        axs[0].set_zlabel('Z')
    axs[0].set_title(alg_type + ' - clustering Plot: '+plt_title)
    # Plot pairwise distances
    for z in range(len(distances_)):
        axs[1].plot(range(len(distances_[z])), distances_[z], marker='o', linestyle='-')
    axs[1].set_title(alg_type + ' - inter cluster distances: '+plt_title)
    axs[1].set_xlabel('Number of Clusters (Point Index)')
    axs[1].set_ylabel('Distance between clusters')
    axs[1].set_ylim(0)
    # Show plot
    plt.tight_layout()
    plt.show()
    return cluster_points


def calculate_cluster_area(cluster_points):
    # Compute convex hull for each cluster
    cluster_areas = []
    cluster_distance = []
    # plt.figure(figsize=(8, 8))
    fig, ax = plt.subplots()
    for c in range(len(cluster_points)):
        cluster_pts = cluster_points[c]
        boundary_polygon = alpha_shape(cluster_pts, 0.5)
        if boundary_polygon != 0:
            cluster_areas.append(boundary_polygon.area)
            # print(f"Area (Alpha shape): {boundary_polygon.area}")
        # Plot the points
        ax.plot(cluster_pts[:, 0], cluster_pts[:, 1], 'o', label='Cluster Points')
        # Plot the boundary polygon
        if boundary_polygon != 0:
            x, y = boundary_polygon.exterior.xy
            ax.plot(x, y, 'r-', label=f'Alpha Shape (alpha={0.5})')

        distances = euclidean_distances(cluster_pts)
        distances_ = []
        for i in range(len(distances)):
            dist = []
            for j in range(len(distances[i])):
                if i != j:
                    dist.append(distances[i][j])
            distances_.extend(dist)
        average_distance = sum(distances_)/len(distances_)
        cluster_distance.append(average_distance)
    plt.show()
    print("Average Area:", sum(cluster_areas)/len(cluster_areas))
    print("Average intra cluster distance: ", sum(cluster_distance)/len(cluster_distance))
    return cluster_areas


if __name__ == "__main__":
    random.seed(87)
