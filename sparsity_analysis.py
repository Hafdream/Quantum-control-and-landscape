import random
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans, DBSCAN
import pandas as pd
from collections import Counter
from scipy.spatial import ConvexHull


# Generate sample points (replace this with your dataset)
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
            dbscan = DBSCAN(eps=eps_init, min_samples=2)
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
    distances = euclidean_distances(cluster_centers)
    distances_ = []
    for i in range(len(distances)):
        dist = []
        for j in range(len(distances[i])):
            if i != j:
                dist.append(distances[i][j])
        distances_.append(dist)

    # print("Cluster centers: ", cluster_centers)
    # print("Distance cal:", distances_)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    colors = np.random.rand(number_of_clusters, 3)
    if pca_result.shape[1] > 2:
        axs[0] = fig.add_subplot(121, projection='3d')
    for i in range(number_of_clusters):
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

    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')
    if pca_result.shape[1] > 2:
        axs[0].set_zlabel('Z')
    axs[0].set_title(alg_type + '-Clustering Plot-'+plt_title)
    # Plot pairwise distances
    for z in range(len(distances_)):
        axs[1].plot(range(len(distances_[z])), distances_[z], marker='o', linestyle='-')
    axs[1].set_title(alg_type + '-Cluster Distances-'+plt_title)
    axs[1].set_xlabel('Point Index')
    axs[1].set_ylabel('Distance')
    axs[1].set_ylim(0)
    # Show plot
    plt.tight_layout()
    plt.show()
    return cluster_points


def calculate_cluster_area(cluster_points):
    # Compute convex hull for each cluster
    cluster_areas = []
    cluster_distance = []
    plt.figure(figsize=(8, 8))
    for c in range(len(cluster_points)):
        cluster_pts = cluster_points[c]
        hull = ConvexHull(cluster_pts)
        # Calculate area of convex hull
        area = hull.volume

        if len(cluster_pts[0]) == 2:
            print("Area of cluster {}: {}".format(c, area))
            plt.scatter(cluster_pts[:, 0], cluster_pts[:, 1], s=10, color='blue', label='Cluster Points')
            plt.xlim(-1, 1)
            plt.ylim(-1, 1)
            # Plot the convex hull
            for simplex in hull.simplices:
                plt.plot(cluster_pts[simplex, 0], cluster_pts[simplex, 1], 'r--', linewidth=2)
            plt.title('Convex Hull around Cluster Points')
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')


        else:
            print("Volume of cluster {}: {}".format(c, area))
        cluster_areas.append(area)

        # pairwise_distances = np.linalg.norm(cluster_pts[:, np.newaxis, :] - cluster_pts[np.newaxis, :, :],
        #                                     axis=-1)
        #
        # # Exclude distances between the same points (diagonal elements)
        # np.fill_diagonal(pairwise_distances, np.nan)
        #
        # # Calculate the average distance between points in the cluster
        # average_distance = np.nanmean(pairwise_distances)
        distances = euclidean_distances(cluster_pts)
        distances_ = []
        for i in range(len(distances)):
            dist = []
            for j in range(len(distances[i])):
                if i != j:
                    dist.append(distances[i][j])
            distances_.extend(dist)
        # print(distances_)
        average_distance = sum(distances_)/len(distances_)
        cluster_distance.append(average_distance)
        print("Average distance between points in the cluster:", average_distance)
    # plt.legend()
    plt.show()
    print("Area sum:", np.sum(cluster_areas))
    print("Average distance(for all clusters): ", sum(cluster_distance)/len(cluster_distance))
    return cluster_areas


if __name__ == "__main__":

    random.seed(87)
    n = 759001  # 6900
    s = 250000  # desired sample size
    skip = sorted(random.sample(range(0, n + 1), n - s))  # the 0-indexed header will not be included in the skip list
    # print("Skip: ", skip)
    x = pd.read_csv("./results/4_param_fidelity_results_25_all.csv")  # , skiprows=skip)
    print(x.shape)
    # print(x.head())
    x_ = x.values.tolist()
    x_f, fid_ = [], []
    a, b, c, d, e = [], [], [], [], []
    for i in range(len(x_)):
        fidelity = x_[i][0]
        if fidelity > 0.95:
            fid_.append(x_[i][0])
            x_f.append(x_[i][1:])
    cluster_p = sparsity_analysis_from_raw_data(np.asarray(x_f), 'dbscan')
    calculate_cluster_area(cluster_p)
