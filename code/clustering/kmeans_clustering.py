from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from clustering.embeddings import EmbeddingsGenerator
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

def cluster_kmeans(embeddings, num_clusters):
    """
    Perform K-Means clustering on embeddings
    Returns: Cluster labels for each argument (array)
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    return cluster_labels

def optimal_k_using_elbow(embeddings, max_k=10):
    """
    Determine the optimal number of clusters using the Elbow Method with Yellowbrick.
    """
    kmeans = KMeans(init='k-means++', n_init=10, max_iter=100, random_state=42)
    visualizer = KElbowVisualizer(kmeans, k=(1, max_k), metric='distortion')
    visualizer.fit(embeddings)
    visualizer.show()


def optimal_k_using_silhouette(embeddings, max_k=10):
    """
    Determine the optimal number of clusters using Silhouette Score with Yellowbrick.
    """
    fig, ax = plt.subplots(3, 2, figsize=(15, 8))  
    ax = ax.flatten() 
    
    plot_index = 0  
    for k in range(2, max_k + 1):
        if plot_index >= len(ax):  
            break 
        
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=100, random_state=42)
        
        visualizer = SilhouetteVisualizer(kmeans, ax=ax[plot_index])  
        visualizer.fit(embeddings)
        ax[plot_index].set_title(f"Silhouette Analysis for k={k}")
        plot_index += 1  

    plt.tight_layout()  
    plt.show()


if __name__ == "__main__":
    embed_gen = EmbeddingsGenerator()
    arguments = [
            "Manchester City dominated possession and controlled the tempo, which ultimately led them to a 3-0 win over Arsenal.",
            "Liverpool’s defense was exposed multiple times by Chelsea's counter-attacks, which led to their 2-1 loss.",
            "Bruno Fernandes assisted Cristiano Ronaldo for the match-winning goal, proving to be the key factor in Manchester United’s 2-1 victory over Tottenham.",
            "Tottenham’s poor form at the start of the season left them with no chance of finishing in the top four.",
            "Leeds United’s relentless attack failed to convert chances, resulting in a frustrating 1-1 draw against West Ham.",
            "Chelsea’s strong defensive unit, led by Thiago Silva and Antonio Rudiger, ensured a 0-0 draw with Liverpool.",
            "Manchester City’s high pressing game forced Leicester into mistakes, which directly contributed to their 4-1 win.",
            "Jordan Pickford’s crucial saves, including a penalty stop, were vital in Everton’s 1-0 win over Aston Villa.",
            "Arsenal’s failure to break down Brighton’s defense led to another disappointing 1-1 draw.",
            "Harry Kane’s brace was the decisive factor in Tottenham’s 3-0 victory over Newcastle.",
            "Southampton’s high defensive line was punished by Manchester United’s counter-attacks, resulting in a 4-2 defeat.",
            "Jarrod Bowen’s pace and creativity helped West Ham secure a 3-1 win over Watford.",
            "Rodri’s control of the midfield was the reason Manchester City dominated Aston Villa 2-0.",
            "Leandro Trossard’s brace was the key to Brighton’s 2-0 victory over Norwich.",
            "Harry Maguire’s defensive errors cost Manchester United dearly in their 3-2 loss to Leicester City."
    ]
    embeddings = embed_gen.get_embeddings(arguments=arguments)
   # optimal_k_using_elbow(embeddings, max_k=10)
    optimal_k_using_silhouette(embeddings, max_k=10)
