import datetime
import json
import pandas as pd
import seaborn as sns
# unused but required import for doing 3d projections with matplotlib < 3.2
import mpl_toolkits.mplot3d  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn import datasets, decomposition
from sklearn.preprocessing import StandardScaler
from skimage import io, color
from sparsity_analysis import sparsity_analysis_kde, sparsity_analysis_from_raw_data,calculate_cluster_area
from umap import UMAP
from sklearn.preprocessing import MinMaxScaler
np.random.seed(5)
from check_repeated_row import count_repeated_rows
import matplotlib.colors as mcolors

def img_compress_svd(image, k):
    if len(image.shape) == 3:
        image_gray = color.rgb2gray(image[:, :, :3])
    else:
        image_gray = image

    U, S, Vt = np.linalg.svd(image_gray, full_matrices=False)

    # Keep only the first k singular values and truncate U and Vt matrices
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    Vt_k = Vt[:k, :]

    # Reconstruct the compressed image
    compressed_image = U_k @ S_k @ Vt_k
    # Plot the original and compressed images

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(image_gray, cmap='gray')
    axes[0].set_title('Original Image')

    axes[1].imshow(compressed_image, cmap='gray')
    axes[1].set_title(f'Compressed Image ({k} singular values)')

    plt.show()
    return compressed_image


def plot_4d(seq_5d):
    x = np.asarray(seq_5d[0])
    y = np.asarray(seq_5d[1])
    z = np.asarray(seq_5d[2])
    a = np.asarray(seq_5d[3])
    b = np.asarray(seq_5d[4])
    # Create a 3D scatter plot with color and size representing additional dimensions
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(y, z, a, c=x, s=b * 10, cmap='jet', vmin=0.95, vmax=1.012)

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('fidelity value')

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()


def plot_overlap_count(seq_, title):
    _count = np.asarray(seq_[0])
    _b = np.asarray(seq_[1])
    _x = np.asarray(seq_[2])
    _y = np.asarray(seq_[3])
    fig = plt.figure(figsize=(12, 8))

    ax = fig.add_subplot()
    ax.set_title(title)
    # original_cmap = plt.cm.viridis
    # inverted_cmap = mcolors.ListedColormap(original_cmap.colors[::-1])
    scatter = ax.scatter(_x, _y, c=_count, cmap='viridis', s=_count*10)

    cbar = plt.colorbar(scatter)
    cbar.set_label('overlap count')
    ax.set_xlabel('a1')
    ax.set_ylabel('a2')
    plt.show()


def plot_2d(seq_3d, fid, title):
    x_ = np.asarray(seq_3d[0])
    y_ = np.asarray(seq_3d[1])
    fig = plt.figure(figsize=(12, 8))

    if len(seq_3d) >= 3:
        z_ = np.asarray(seq_3d[2])
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(title)
        scatter = ax.scatter(x_, y_, z_, c=fid, cmap='jet', s=20, vmin=0.0, vmax=1.0)
        cbar = plt.colorbar(scatter)
        cbar.set_label('fidelity value')

        # Set axis labels
        ax.set_zlabel('a3')
    else:
        ax = fig.add_subplot()
        ax.set_title(title)
        scatter = ax.scatter(x_, y_, c=fid, cmap='jet', s=20, vmin=0.0, vmax=1.0)
        # # ######################################################################
        # grid_resolution = len(x_)  # adjust based on your preference
        # grid_x = np.linspace(min(x_), max(x_), grid_resolution)
        # grid_y = np.linspace(min(y_), max(y_), grid_resolution)
        # grid_values = np.zeros((grid_resolution, grid_resolution))
        #
        # # Assign values to the grid based on scatter plot data
        # i_ = 0
        # for xi, yi in zip(x_, y_):
        #     grid_x_index = np.argmin(np.abs(grid_x - xi))
        #     grid_y_index = np.argmin(np.abs(grid_y - yi))
        #     grid_values[grid_y_index, grid_x_index] += fid[i_]
        #     i_ += 1
        #
        # # Plot using imshow
        # plt.imshow(grid_values, extent=(min(x_), max(x_), min(y_), max(y_)), origin='lower', cmap='jet')
        #
        # # ######################################################################
        cbar = plt.colorbar(scatter)
        cbar.set_label('fidelity value')
    ax.set_xlabel('a1')
    ax.set_ylabel('a2')
    # Show the plot
    plt.show()


def run_tsne(X, n_components):
    print("fitting t-SNE")
    tsne = TSNE(n_components=n_components, random_state=42)
    X_tsne = tsne.fit_transform(np.asarray(X))
    # X_tsne_scaled = 2 * (X_tsne - np.min(X_tsne, axis=0)) / np.ptp(X_tsne, axis=0) - 1

    return X_tsne


def run_pca(X, numComponents):
    print("Fitting PCA ")
    pca = decomposition.PCA(n_components=numComponents)
    # pca.fit(X)
    # Step 1: Apply PCA for dimensionality reduction
    # scaler = StandardScaler()
    # data_scaled = scaler.fit_transform(X)
    pca_result = pca.fit_transform(X)
    # inv_rec = pca.inverse_transform(X)
    # # Step 2: Explained Variance
    explained_variance_ratio = pca.explained_variance_ratio_

    # Plot explained variance
    # plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
    # plt.xlabel('Principal Component')
    # plt.ylabel('Explained Variance Ratio')
    # plt.title('Cumulative Explained Variance')
    # plt.show()

    # # Step 3: Create a scatter plot of the first x principal components
    # principal_df = pd.DataFrame(data=pca_result[:, :2], columns=['PC1', 'PC2'])
    # principal_df['target'] = fid_
    #
    # plt.figure(figsize=(8, 6))
    # sns.scatterplot(data=principal_df, x='PC1', y='PC2', hue='target', palette='icefire', s=20)
    # plt.title('2D PCA Scatter Plot')
    # plt.show()

    # Step 4: Cluster Analysis (Visual Inspection)

    # Step 5: Correlation with Original Dimensions
    original_dimensions_correlation = np.corrcoef(np.asarray(X).T, pca_result.T[:2])
    # print("Correlation with Original Dimensions:", original_dimensions_correlation)

    # Step 6: Contribution of Original Dimensions
    loadings = pca.components_
    print("Loadings (Contribution of Original Dimensions):\n", loadings)
    # np.save('./results/4param_PCA_loadings.npy', loadings)
    # Step 7: Interpretation of Clusters or Patterns (Visual Inspection)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(pca_result)
    # return scaled_data
    return pca_result


def run_umap(X, numComponents):
    # umap = UMAP(n_neighbors=10, min_dist=1, n_components=numComponents)
    umap = UMAP(n_neighbors=5, min_dist=1, n_components=numComponents)
    X_umap = umap.fit_transform(X)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(X_umap)
    return scaled_data


if __name__ == "__main__":
    import random

    random.seed(87)
    n = 6250000  # 1000000  #  759001  # 6900
    s = 625000  # 525000 # desired sample size
    skip = sorted(random.sample(range(0, n + 1), n - s))  # the 0-indexed header will not be included in the skip list
    # print("Skip: ", skip)
    # x = pd.read_csv("./results/3_param_fidelity_T2_results_50_exponential.csv")  # , skiprows=skip)
    # x = pd.read_csv("./results/2_param_fidelity_results_100_all.csv")  # , skiprows=skip)
    # x = pd.read_csv("./results/2_param_fidelity_T2_results_50_sinusoidal.csv")
    # x = pd.read_csv("./results/3_param_fidelity_T2_results_50_sinusoidal_(seq[0]+seq[1]*(delta_t)+seq[2]*sin(a3_values[i]).csv")  # , skiprows=skip)
    # x = pd.read_csv("./results/4_param_fidelity_T2_results_50_sinusoidal.csv", skiprows=skip)

    # Using random pulses
    # x = pd.read_csv("./results/2_param_fidelity_results_100_all.csv")
    # x = pd.read_csv("./results/3_param_fidelity_results_100_all.csv")#, skiprows=skip)
    # x = pd.read_csv("./results/4_param_fidelity_results.csv")
    # x = pd.read_csv("./results/2param_PCA_SGD_from_loadings.csv")
    # x = pd.read_csv("./results/3param_PCA_SGD_from_loadings2.csv")
    x = pd.read_csv("./results/4param_dqn_pca_loadings2.csv")
    threshold = 0.01
    repeated_rows_count, actual_values = count_repeated_rows(x, threshold, True)
    sorted_result = sorted(repeated_rows_count.items(), key=lambda val: val[1], reverse=True)

    print("Counts of repeated pulses:")
    tot_count = 0
    x_with_count = []
    x_count_, x_count_fid, x_count1, x_count2 = [], [], [], []
    for row, count in sorted_result:  # [:100]:
        tot_count += count
        x_with_count.append([count, row[0], row[1], row[2]])
        print("val -> count: ", row, "->", count, "\t", "| \tactual values: ", actual_values[row])
    #     x_count_.append(count)
    #     x_count_fid.append(row[0])
    #     x_count1.append(row[1])
    #     x_count2.append(row[2])
    #
    # x_with_count.append(x_count_)
    # x_with_count.append(x_count_fid)
    # x_with_count.append(x_count1)
    # x_with_count.append(x_count2)
    x_with_count = np.asarray(x_with_count).T

    print("Len of repeated: ", len(sorted_result), "\t Sum of repeated: ", tot_count)
    print(x.shape)
    # print(x.head())
    x_ = x.values.tolist()
    x_f, x_bdr, fid_ = [], [], []
    a, b, c, d, e = [], [], [], [], []
    a_, b_, c_ = [], [], []
    for i in range(len(x_)):
        if x_[i][0] > 0.95:
            x_bdr.append(x_[i][1:])
            a_.append(x_[i][1])
            b_.append(x_[i][2])
            if len(x_[i]) >= 4:
                c_.append(x_[i][3])
        fid_.append(x_[i][0])
        x_f.append(x_[i][1:])
        a.append(x_[i][1])
        b.append(x_[i][2])
        if len(x_[i]) >= 4:
            c.append(x_[i][3])
        if len(x_[i]) >= 5:
            d.append(x_[i][4])
        if len(x_[i]) >= 6:
            e.append(x_[i][4])

    num_components = 2
    res = run_pca(x_f, num_components)
    # print("Fitting UMAP")
    # res = run_umap(x_f, num_components)
    # res_tsne = run_tsne(np.asarray(x_f))
    date_str = (str(datetime.datetime.now()).replace(":", "")
                .replace(" ", "").replace("-", ""))
    res_list = res.tolist()

    x_a, x_b, x_c = [], [], []
    x_f_all, x_f1, x_f2, x_f3, x_fid = [], [], [], [], []
    for z in range(len(res_list)):
        x_a.append(res_list[z][0])
        x_b.append(res_list[z][1])
        if num_components == 3:
            x_c.append(res_list[z][2])
        if fid_[z] > 0.95:
            x_f_all.append(res_list[z])
            x_f1.append(res_list[z][0])
            x_f2.append(res_list[z][1])
            x_fid.append(fid_[z])
            if num_components == 3:
                x_f3.append(res_list[z][2])
        # print("X_F: ", x_f[z], "\n\tPCA: ", res_list[z])
    if num_components == 3:
        seq_3d = [x_a, x_b, x_c]
        seq_3d_f = [x_f1, x_f2, x_f3]
    else:
        seq_3d = [x_a, x_b]
        seq_3d_f = [x_f1, x_f2]

    seq_5d = []
    for jj, x in enumerate([a, b, c, d, e]):
        if len(x) > 0:
            seq_5d.append(x)
            print("Len: ", len(seq_5d[jj]))
    seq_5d_ = []
    for xx in [a_, b_, c_]:
        if len(xx) > 0:
            seq_5d_.append(xx)

    plot_overlap_count(x_with_count, "4 param: Pulse Overlap Count\n(color → overlap count, size → overlap count)")
    plot_2d(seq_5d, fid_, "4 param: PPO raw data from PCA loadings(Only 2|3 Dimensions)")
    plot_2d(seq_5d_, x_fid, "2 param: PPO raw data from PCA loadings at 95% fid(Only 2|3 Dimensions)")
    plot_2d(seq_3d, fid_, "2 param: after applying PCA")
    plot_2d(seq_3d_f, x_fid, "4 param: after applying PCA at 95% fid")

    # sparsity_analysis_kde(x_f, x_f2, 0.4, 4)
    print("Running sparsity analysis")
    print(np.asarray(x_f_all).shape, " | ", np.asarray(x_bdr).shape)

    cluster_p = sparsity_analysis_from_raw_data(np.asarray(x_f_all), "dbscan", "after PCA")
    calculate_cluster_area(cluster_p)

    cluster_pbdr = sparsity_analysis_from_raw_data(np.asarray(x_bdr), "dbscan", "before DR(Only 2/3 Dimensions)")
    # calculate_cluster_area(cluster_pbdr)

