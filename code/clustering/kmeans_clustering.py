from sklearn.cluster import KMeans
import numpy as np

def cluster_kmeans(embeddings, num_clusters):
    """
    Perform K-Means clustering on embeddings.
    
    Parameters:
        embeddings (numpy.ndarray): Array of Sentence-BERT embeddings.
        num_clusters (int): Number of clusters for K-Means.
    
    Returns:
        cluster_labels (numpy.ndarray): Cluster labels for each argument.
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    return cluster_labels

if __name__ == "__main__":
    # Example usage
    # Load your embeddings here
    example_embeddings = np.random.rand(100, 768)  # Example: 100 arguments with 768-dim embeddings
    num_clusters = 5
    labels = cluster_kmeans(example_embeddings, num_clusters)
    print("K-Means Cluster Labels:", labels)
