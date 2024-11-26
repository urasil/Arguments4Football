from kmeans_clustering import cluster_kmeans
from entity_clustering import cluster_by_entities
from topic_clustering import cluster_by_topics
from dbscan_clustering import cluster_dbscan
from embeddings import EmbeddingsGenerator

class Clustering:
    def __init__(self, method, sentences, num_clusters=5, eps=0.3, min_samples=5):

        self.method = method
        self.num_clusters = num_clusters
        self.sentences = sentences
        self.sentence_embeddings = EmbeddingsGenerator.get_embeddings(sentences=sentences)
        self.eps = eps
        self.min_samples = min_samples

    def cluster(self):
        """
        Performs the clustering of choice
        """
        if self.method == "kmeans":
            return cluster_kmeans(embeddings=self.sentence_embeddings, num_clusters=self.num_clusters)
        # NEED TO FIX ENTITY CODE AND LOGIC
        elif self.method == "entity":
            return cluster_by_entities(embeddings=self.sentence_embeddings, sentences=self.sentences, num_clusters=self.num_clusters)
        elif self.method == "bertopic":
            topics, _ = cluster_by_topics(embeddings=self.sentence_embeddings)
            return topics
        elif self.method == "dbscan":
            return cluster_dbscan(embeddings=self.sentence_embeddings, eps=self.eps, min_samples=self.min_samples)
        else:
            raise ValueError(f"Unsupported clustering method: {self.method}")



if __name__ == "__main__":
    arguments = [
            "Player X's goal in the final minute proves his incredible composure under pressure. It was the decisive moment of the match.",
            "Team Y's defensive strategy failed because they left gaps in the midfield. This allowed Player Z to exploit the space and score.",
            "The referee made a controversial decision, awarding a penalty to Team A that changed the outcome of the game.",
            "Player B's performance was outstanding as he scored a hat-trick, carrying his team to victory.",
            "If the weather conditions had been better, Team C might have performed more effectively in the match.",
            "Team D dominated possession but failed to convert their chances, highlighting their inefficiency in front of the goal."
        ]

    for method in ["kmeans", "entity", "bertopic", "dbscan"]:
        print(f"\nClustering using {method.upper()}:")
        clustering = Clustering(method=method, sentences=arguments, num_clusters=5, eps=0.5, min_samples=5)
        labels = clustering.cluster()
        print(f"Cluster Labels for {method.upper()}: {labels}")