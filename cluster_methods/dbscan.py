from cluster_methods.base import ClusterBase
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


import numpy as np

from sklearn import metrics
from sklearn.cluster import DBSCAN

# db = DBSCAN(eps=0.9, min_samples=10).fit(X)
# labels = db.labels_

# # Number of clusters in labels, ignoring noise if present.
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# n_noise_ = list(labels).count(-1)

# print("Estimated number of clusters: %d" % n_clusters_)
# print("Estimated number of noise points: %d" % n_noise_)

# # print(f"Homogeneity: {metrics.homogeneity_score(labels_true, labels):.3f}")
# # print(f"Completeness: {metrics.completeness_score(labels_true, labels):.3f}")
# # print(f"V-measure: {metrics.v_measure_score(labels_true, labels):.3f}")
# # print(f"Adjusted Rand Index: {metrics.adjusted_rand_score(labels_true, labels):.3f}")
# # print(
# #     "Adjusted Mutual Information:"
# #     f" {metrics.adjusted_mutual_info_score(labels_true, labels):.3f}"
# # )

# unique_labels = set(labels)
# core_samples_mask = np.zeros_like(labels, dtype=bool)
# core_samples_mask[db.core_sample_indices_] = True

# nrows, ncols = 10, 10
# fig, axes = plt.subplots(
#     nrows=nrows, ncols=ncols, figsize=(10, 10), constrained_layout=True
# )
# fig.suptitle(f"Estimated number of clusters: {n_clusters_}", fontsize=16)
# colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]


# colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
# for i in range(nrows):
#     for j in range(ncols):
#         ax = axes[i, j]
#         if i == j:
#             continue
#     for k, col in zip(unique_labels, colors):
#         if k == -1:
#             # Black used for noise.
#             col = [0, 0, 0, 1]

#         class_member_mask = labels == k

#         xy = X[class_member_mask & core_samples_mask]
#         ax.plot(
#             xy[:, j],
#             xy[:, i],
#             "o",
#             markerfacecolor=tuple(col),
#             markeredgecolor="k",
#             markersize=14,
#         )

#         xy = X[class_member_mask & ~core_samples_mask]
#         ax.plot(
#             xy[:, j],
#             xy[:, i],
#             "o",
#             markerfacecolor=tuple(col),
#             markeredgecolor="k",
#             markersize=6,
#         )
#         ax.set_xlabel(f"Feature {j}", fontsize=8)
#         ax.set_ylabel(f"Feature {i}", fontsize=8)
#         ax.tick_params(axis="both", which="major", labelsize=6)

# plt.title(f"Estimated number of clusters: {n_clusters_}")
# plt.show()


def get_dbscan_clusters(X, eps=0.9, min_samples=10):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    return labels, n_clusters_, n_noise_

class DBSCANCluster(ClusterBase):
    def __init__(self, eps=0.9, min_samples=10, columns=None):
        self.eps = eps
        self.min_samples = min_samples
        self.db = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.trained = False
        self.columns = columns

    def fit(self, X, verbose=True):
        if self.trained:
            raise ValueError("DBSCAN has already been trained. Please create a new instance to fit again.")
        self.db.fit(X)
        self.labels = self.db.labels_
        self.n_clusters_ = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        self.n_noise_ = list(self.labels).count(-1)
        
        if self.n_clusters_ == 0:
            if verbose:
                print("No clusters found. Please check your data and parameters.")
            sil = -1
        else:
            try:
                sil = metrics.silhouette_score(X, self.labels)
            except:
                sil = -1
                if verbose:
                    print("Silhouette score could not be computed. Please check your data and parameters.")
        
        self.trained = True
        if verbose:
            print(f"DBSCANCluster Silhouette Coefficient: {sil:.3f}")
        return self.labels, self.n_clusters_, self.n_noise_, sil
    
    def predict(self, X):
        if not self.trained:
            raise ValueError("DBSCAN has not been trained yet. Please fit the model first.")
        return self.db.predict(X)
    
    def plot(self, X, nrows=10, ncols=10):
        unique_labels = set(self.labels)
        core_samples_mask = np.zeros_like(self.labels, dtype=bool)
        core_samples_mask[self.db.core_sample_indices_] = True

        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(50, 50), constrained_layout=True
        )
        fig.suptitle(f"Estimated number of clusters: {self.n_clusters_}", fontsize=16)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

        for i in range(nrows):
            for j in range(ncols):
                ax = axes[i, j]
                if i == j:
                    continue
                for k, col in zip(unique_labels, colors):
                    if k == -1:
                        # Black used for noise.
                        col = [0, 0, 0, 1]
                        continue

                    class_member_mask = self.labels == k

                    xy = X[class_member_mask & core_samples_mask]
                    ax.plot(
                        xy[:, j],
                        xy[:, i],
                        "o",
                        markerfacecolor=tuple(col),
                        markeredgecolor="k",
                        markersize=14,
                    )

                    xy = X[class_member_mask & ~core_samples_mask]
                    ax.plot(
                        xy[:, j],
                        xy[:, i],
                        "o",
                        markerfacecolor=tuple(col),
                        markeredgecolor="k",
                        markersize=6,
                    )
                    ax.set_xlabel(self.columns[j], fontsize=8)
                    ax.set_ylabel(self.columns[i], fontsize=8)
                    ax.tick_params(axis="both", which="major", labelsize=6)

        plt.title(f"Estimated number of clusters: {self.n_clusters_}")
        plt.show()

