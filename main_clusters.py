
from tm_data.preprocessing import preprocess_tm_data
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from cluster_methods.dbscan import DBSCANCluster

if __name__ == "__main__":
    # Example usage
    
    csv_path = "/Users/arthurtestard/LLM-UQ-with-TM/models/fetched_training_data.csv"
    X, columns = preprocess_tm_data(csv_path, binarize=False, drop_epoch=True, return_columns=True) # [:, :10]
    X = StandardScaler().fit_transform(X)

    dbscan_cluster = DBSCANCluster(eps=0.7, min_samples=10, columns=columns)
    labels, n_clusters_, n_noise_, sil = dbscan_cluster.fit(X)
    print(f"Estimated number of clusters: {n_clusters_}")
    print(f"Estimated number of noise points: {n_noise_}")
    dbscan_cluster.plot(X)