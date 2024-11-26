import spacy
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

def extract_entities(arguments):
    """
    Extract entities from a list of arguments using spaCy.  
    Returns: List of extracted entities from arguments (list of str)
    """
    nlp = spacy.load("en_core_web_sm")
    entities_list = []
    for arg in arguments:
        doc = nlp(arg)
        entities = [ent.text for ent in doc.ents]
        entities_list.append(" ".join(entities))
    return entities_list

def cluster_by_entities(arguments, num_clusters):
    """
    Cluster arguments based on extracted entities.
    Returns: cluster_labels: Cluster labels for each argument (list of int)
    """
    # Step 1: Extract entities
    entities_list = extract_entities(arguments)
    
    # Step 2: Encode entities using Sentence-BERT or TF-IDF
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(entities_list)
    
    # Step 3: Cluster embeddings using K-Means
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    return cluster_labels

if __name__ == "__main__":
    
    arguments = [
        "Player X scored a great goal in the match.",
        "Team Y's defense strategy was ineffective.",
        "Player Z was injured and missed the game.",
        "The referee's decision was controversial.",
        "Team A's midfield was dominant throughout the game."
    ]
    num_clusters = 3
    labels = cluster_by_entities(arguments, num_clusters)
    print("Cluster Labels by Entities:", labels)
