from bertopic import BERTopic
from umap import UMAP
import math
from hdbscan import HDBSCAN


def cluster_by_topics(arguments):
    """
    Cluster arguments based on latent topics using BERTopic.
    
    Returns: Topic labels and topic_info
    """
    if len(arguments) < 2:
        raise ValueError("At least 2 data points are required for clustering.")
    
    # Dynamically adjust n_neighbors and min_samples based on data size
    nei = max(2, min(len(arguments) - 1, len(arguments) // 5))  # Ensure valid n_neighbors
    min_samples = min(len(arguments) - 1, 10)  # Ensure min_samples does not exceed data points

    # Configure BERTopic with UMAP and adjusted HDBSCAN
    topic_model = BERTopic(
        umap_model=UMAP(n_neighbors=nei, n_components=2, min_dist=0.1, random_state=31),
        hdbscan_model=HDBSCAN(min_samples=min_samples, gen_min_span_tree=True),
    )

    topics, probs = topic_model.fit_transform(arguments)
    topic_info = topic_model.get_topic_info()

    return topics, topic_info

if __name__ == "__main__":
    # Example usage
    arguments = [
        "Player X scored a great goal in the match.",
        "Team Y's defense strategy was ineffective.",
        "Player Z was injured and missed the game.",
        "The referee's decision was controversial.",
        "Team A's midfield was dominant throughout the game."
    ]
    topics, topic_info = cluster_by_topics(arguments)
    print("Topic Labels by BERTopic:", topics)
    print("Topic Information:\n", topic_info)
