from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def load_model_and_tokenizer(model_name):
    """
    Load the transformers model and tokenizer.
    Returns: fine-tuned model and tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return model, tokenizer

def identify_arguments(model, tokenizer, sentences, threshold=0.5):
    """
    Identify the arguments in a list of sentences
    Returns: list(str) = arguments
    """
    model.eval()  # Set model to evaluation mode
    sentences = [sentence.lower() for sentence in sentences]
    arguments = []

    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

    for i, prob in enumerate(probabilities):
        if prob[1] > threshold:  # If probability for "argument" is greater than threshold
            arguments.append(sentences[i])
    
    return arguments

if __name__ == "__main__":
    # Example usage
    model_name = "curr-pure-best"
    sentences = [
        "Player X is one of the best forwards in the league.",
        "The weather tomorrow is expected to be sunny.",
        "Team Y's tactics were ineffective in the second half.",
        "This year, the company achieved record profits.",
        "The referee made a controversial decision that influenced the game."
    ]
    
    model, tokenizer = load_model_and_tokenizer(model_name)
    identified_arguments = identify_arguments(model, tokenizer, sentences)
    print("Identified Arguments:")
    for arg in identified_arguments:
        print(f"- {arg}")
