from bertopic import BERTopic

def cluster_by_topics(arguments):
    """
    Cluster arguments based on latent topics using BERTopic.
    
    Parameters:
        arguments (list of str): List of textual arguments.
    
    Returns:
        topics (list of int): Topic labels for each argument.
        topic_info (pd.DataFrame): Information about the topics and their keywords.
    """
    # Step 1: Initialize BERTopic model
    topic_model = BERTopic()

    # Step 2: Fit the model on the arguments
    topics, probs = topic_model.fit_transform(arguments)
    
    # Step 3: Get topic information
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
