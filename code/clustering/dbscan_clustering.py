from sklearn.cluster import DBSCAN
import numpy as np

def cluster_dbscan(embeddings, eps=0.5, min_samples=5):
    """
    Perform DBSCAN clustering on embeddings
    Returns: Cluster labels for each argument (array)
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    cluster_labels = dbscan.fit_predict(embeddings)
    return cluster_labels

if __name__ == "__main__":

    example_embeddings = np.random.rand(100, 768)  # 100 arguments with 768-dim embeddings
    eps_value = 0.3
    min_samples_value = 5
    labels = cluster_dbscan(example_embeddings, eps=eps_value, min_samples=min_samples_value)
    print("DBSCAN Cluster Labels:", labels)
