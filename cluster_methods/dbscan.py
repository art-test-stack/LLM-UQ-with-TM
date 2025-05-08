from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from tm_data.preprocessing import preprocess_tm_data

# centers = [[1, 1], [-1, -1], [1, -1]]
# X, labels_true = make_blobs(
#     n_samples=750, centers=centers, cluster_std=0.4, random_state=0
# )

csv_path = "/Users/arthurtestard/LLM-UQ-with-TM/models/fetched_training_data.csv"

X = preprocess_tm_data(csv_path, binarize=False, drop_epoch=True)

X = StandardScaler().fit_transform(X)


print("X.shape", X.shape)
import numpy as np

from sklearn import metrics
from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.9, min_samples=10).fit(X)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)

# print(f"Homogeneity: {metrics.homogeneity_score(labels_true, labels):.3f}")
# print(f"Completeness: {metrics.completeness_score(labels_true, labels):.3f}")
# print(f"V-measure: {metrics.v_measure_score(labels_true, labels):.3f}")
# print(f"Adjusted Rand Index: {metrics.adjusted_rand_score(labels_true, labels):.3f}")
# print(
#     "Adjusted Mutual Information:"
#     f" {metrics.adjusted_mutual_info_score(labels_true, labels):.3f}"
# )
print(f"Silhouette Coefficient: {metrics.silhouette_score(X, labels):.3f}")

unique_labels = set(labels)
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

nrows, ncols = 10, 10
fig, axes = plt.subplots(
    nrows=nrows, ncols=ncols, figsize=(10, 10), constrained_layout=True
)
fig.suptitle(f"Estimated number of clusters: {n_clusters_}", fontsize=16)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]


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

        class_member_mask = labels == k

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
        ax.set_xlabel(f"Feature {j}", fontsize=8)
        ax.set_ylabel(f"Feature {i}", fontsize=8)
        ax.tick_params(axis="both", which="major", labelsize=6)

plt.title(f"Estimated number of clusters: {n_clusters_}")
plt.show()