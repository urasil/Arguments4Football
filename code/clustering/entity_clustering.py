import spacy
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

def extract_and_normalize_entities(arguments, nlp):
    """
    Extract and normalize entities from arguments using spaCy.
    Maps entities to their canonical form.
    Returns: List of normalized entities.
    """
    entities_list = []
    for arg in arguments:
        doc = nlp(arg)
        entities = [ent.text.lower() for ent in doc.ents]  # Normalize case
        entities_list.append(" ".join(entities))
    return entities_list

def cluster_sentences_with_entities(arguments, num_clusters):
    """
    Cluster arguments based on sentence embeddings and extracted entities.
    Returns: cluster_labels: Cluster labels for each argument (list of int)
    """
    # Load spaCy model for entity recognition
    nlp = spacy.load("en_core_web_sm")
    
    # Step 1: Extract normalized entities
    normalized_entities = extract_and_normalize_entities(arguments, nlp)
    
    # Step 2: Compute embeddings for both sentences and entities
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentence_embeddings = model.encode(arguments)
    entity_embeddings = model.encode(normalized_entities)
    
    # Step 3: Combine embeddings (simple concatenation here)
    combined_embeddings = [
        list(sent_emb) + list(ent_emb) 
        for sent_emb, ent_emb in zip(sentence_embeddings, entity_embeddings)
    ]
    
    # Step 4: Apply clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(combined_embeddings)
    return cluster_labels

if __name__ == "__main__":
    arguments = [
        "Player X scored a great goal in the match.",
        "Team Y's defense strategy was ineffective.",
        "Player Z was injured and missed the game.",
        "The referee's decision was controversial.",
        "Team A's midfield was dominant throughout the game.",
        "Manchester United scored a dramatic late goal.",
        "Red Devils secured the win in injury time."
    ]
    num_clusters = 3
    labels = cluster_sentences_with_entities(arguments, num_clusters)
    print("Cluster Labels by Entities:", labels)
