import datetime
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import decomposition
from skimage import color
from sparsity_analysis import sparsity_analysis_from_raw_data, calculate_cluster_area
from umap import UMAP
from sklearn.preprocessing import MinMaxScaler
np.random.seed(5)
from check_repeated_row import count_repeated_rows
import textwrap


def plot_overlap_count(seq_, title):
    _count = np.asarray(seq_[0])
    _b = np.asarray(seq_[1])
    _x = np.asarray(seq_[2])
    _y = np.asarray(seq_[3])
    fig = plt.figure(figsize=(5, 4))

    ax = fig.add_subplot()
    wrapped_title = "\n".join(textwrap.wrap(title, width=25))
    ax.set_title(wrapped_title, fontsize=18)
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_yticks([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])

    scatter = ax.scatter(_x, _y, c=_count, cmap='viridis', s=_count*10)

    cbar = plt.colorbar(scatter)
    cbar.set_label('overlap count', fontsize=18)
    cbar.ax.tick_params(labelsize=14)

    ax.set_xlabel('pc1', fontsize=18)
    ax.set_ylabel('pc2', fontsize=18)

    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_yticks([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])

    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    plt.show()


def plot_2d(seq_3d, fid, title):
    x_ = np.asarray(seq_3d[0])
    y_ = np.asarray(seq_3d[1])
    fig = plt.figure(figsize=(5, 4))

    ax = fig.add_subplot()
    wrapped_title = "\n".join(textwrap.wrap(title, width=25))
    ax.set_title(wrapped_title, fontsize=18)
    scatter = ax.scatter(x_, y_, c=fid, cmap='jet', s=25, vmin=0.0, vmax=1.0)
    # scatter = ax.scatter(x_, y_, c=fid, cmap='jet', s=5, vmin=0.0, vmax=1.0)
    cbar = plt.colorbar(scatter)
    cbar.set_label('fidelity value', fontsize=18)
    # cbar.ax.set_position([4.95, 0.1, 0.3, 3.8])
    # cbar.ax.tick_params(labelsize=14)
    ax.set_xlabel('pc1', fontsize=18)
    ax.set_ylabel('pc2', fontsize=18)
    # ax.set_xticklabels([-1.0, -0.5, 0.0, 0.5, 1.0], fontsize=11)
    # ax.set_yticklabels([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0], fontsize=11)
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_yticks([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
    # ax.set_ylim(top=None, bottom=None)
    # ax.set_xlim(left=None, right=None)
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    # Show the plot
    plt.show()


def run_tsne(X, n_components):
    print("fitting t-SNE")
    tsne = TSNE(n_components=n_components, random_state=42)
    X_tsne = tsne.fit_transform(np.asarray(X))

    return X_tsne

def run_umap(X, numComponents):
    umap = UMAP(n_neighbors=5, min_dist=1, n_components=numComponents)
    X_umap = umap.fit_transform(X)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(X_umap)
    return scaled_data


def run_pca(X, numComponents):
    print("Fitting PCA ")
    pca = decomposition.PCA(n_components=numComponents)

    pca_result = pca.fit_transform(X)
    explained_variance_ratio = pca.explained_variance_ratio_
    print(f"Variance Ratio: {explained_variance_ratio}")

    loadings = pca.components_
    # np.save('./results/4param_PCA_loadings.npy', loadings)
    return pca_result


if __name__ == "__main__":
    import random
    random.seed(87)
    n = 6250000  # 1000000  #  759001  # 6900
    s = 625000  # 525000 # desired sample size
    skip = sorted(random.sample(range(0, n + 1), n - s))  # the 0-indexed header will not be included in the skip list

    # Using Brute-force pulses
    # x = pd.read_csv("./results/2_param_fidelity_results_100_all.csv")
    # x = pd.read_csv("./results/3_param_fidelity_results_100_all.csv")  #, skiprows=skip)
    # x = pd.read_csv("./results/4_param_fidelity_results.csv")  # only high fid values

    # Using SGD,GA,RL Algorithms
    # x = pd.read_csv("./results/2param_SGD_from_PCA_loadings_20240917.csv")
    # x = pd.read_csv("./results/3param_SGD_from_PCA_loadings_20240917.csv")
    # x = pd.read_csv("./results/4param_SGD_from_PCA_loadings-20240917.csv")

    # x = pd.read_csv("./results/2param_ga_pca_loadings_20240922.csv")
    # x = pd.read_csv("./results/3param_ga_pca_loadings_20240922.csv")
    # x = pd.read_csv("./results/4param_ga_pca_loadings_20240922.csv")

    # x = pd.read_csv("./results/2param_QL_e_greedy_0p9_from_PCA_loadings_with_experience_20240527.csv")
    # x = pd.read_csv("./results/3param_QL_e_greedy_0p9_from_PCA_loadings_20240917.csv")
    # x = pd.read_csv("./results/4param_QL_e_greedy_0p9_from_PCA_loadings_20240917.csv")

    # x = pd.read_csv("./results/2param_dqn_pca_loadings_20240529.csv")
    # x = pd.read_csv("./results/3param_dqn_pca_loadings_20240917.csv")
    # x = pd.read_csv("./results/4param_dqn_pca_loadings20240530.csv")

    x = pd.read_csv("./results/2param_PPO_rand_initialization_pca_loadings_20240527.csv")
    # x = pd.read_csv("./results/3param_PPO_rand_initialization_pca_loadings_20240527.csv")
    # x = pd.read_csv("./results/4param_PPO_rand_initialization_pca_loadings_20240916_4.csv")

    threshold = 0.1
    repeated_rows_count, actual_values = count_repeated_rows(x, threshold, True)
    sorted_result = sorted(repeated_rows_count.items(), key=lambda val: val[1], reverse=True)

    print("Counts of repeated pulses:")
    tot_count = 0
    x_with_count = []
    for row, count in sorted_result:  # [:100]:
        tot_count += count
        x_with_count.append([count, row[0], row[1], row[2]])
        # print("val -> count: ", row, "->", count, "\t", "| \tactual values: ", actual_values[row])

    x_with_count = np.asarray(x_with_count).T

    print("Len of repeated: ", len(sorted_result), "\t Sum of repeated: ", tot_count)
    print(x.shape)

    two_param_analysis = True  # True if current analysis is for 2 param pulse
    num_components = 2

    x_ = x.values.tolist()
    x_pulse, fid_ = [], []
    a1_2param, a2_2param = [], []

    for i in range(len(x_)):
        fid_.append(x_[i][0])
        x_pulse.append(x_[i][1:])

        if two_param_analysis:
            a1_2param.append(x_[i][1])
            a2_2param.append(x_[i][2])

    res = run_pca(x_pulse, num_components)
    date_str = (str(datetime.datetime.now()).replace(":", "")
                .replace(" ", "").replace("-", ""))
    res_list = res.tolist()

    if not two_param_analysis:
        x_pc1, x_pc2, x_pc3 = [], [], []  # Principal components (after PCA)
        x_high_fidelity_pulse = []
        for z in range(len(res_list)):
            x_pc1.append(res_list[z][0])
            x_pc2.append(res_list[z][1])

            if num_components == 3:
                x_pc3.append(res_list[z][2])

            if fid_[z] > 0.95:
                x_high_fidelity_pulse.append(res_list[z])

        if num_components == 2:
            seq_3d = [x_pc1, x_pc2]
        else:
            seq_3d = [x_pc1, x_pc2, x_pc3]
        plot_overlap_count(x_with_count, "3 param: GA Overlap count")
        plot_2d(seq_3d, fid_, "3 param: GA after applying PCA")

        # Find clusters and calculate cluster area
        cluster_p = sparsity_analysis_from_raw_data(np.asarray(x_high_fidelity_pulse),
                                                    "dbscan", "DQN(3 param)")
        calculate_cluster_area(cluster_p)

    else:
        seq_2param = []
        for jj, x in enumerate([a1_2param, a2_2param]):
            if len(x) > 0:
                seq_2param.append(x)
                print("Len: ", len(seq_2param[jj]))
        plot_2d(seq_2param, fid_, "2 param:  GA")

        # Find clusters and calculate cluster area
        x_high_fidelity_pulse_2param = []
        for z in range(len(res_list)):
            if fid_[z] > 0.95:
                x_high_fidelity_pulse_2param.append(x_pulse[z])
        cluster_p = sparsity_analysis_from_raw_data(np.asarray(x_high_fidelity_pulse_2param),
                                                    "dbscan", "DQN(3 param)")
        calculate_cluster_area(cluster_p)




