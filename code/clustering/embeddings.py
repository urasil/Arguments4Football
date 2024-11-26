from transformers import AutoModel, AutoTokenizer, RobertaForSequenceClassification, RobertaTokenizer
import torch
"""
In a transformer model like RoBERTa or BERT:

Hidden states are the intermediate representations produced by each layer of the model for each token in the input sequence
Each layer refines the token representations based on the context learned from the input. These are represented as tensors of shape [batch_size, seq_len, hidden_dim]
batch_size: Number of input sequences in the batch
seq_len: Length of the tokenized input sequence
hidden_dim: Dimensionality of the token representations (768 for RoBERTa base)
The last layer's hidden states refer to the token embeddings from the final layer of the transformer. These embeddings capture the most refined and contextualized information about each token, as learned by the model
"""

class EmbeddingsGenerator:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device=None, pooling_type="mean-pooling"):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.pooling_type = pooling_type

        if "sentence-transformers" in model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
        else:  
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            self.model = RobertaForSequenceClassification.from_pretrained(model_name).to(self.device)

    def get_embeddings(self, sentences):
        """
        Compute embeddings for a list of sentences
        Returns: List[List[int]] - List[Embeddings]
        """

        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # Unsupervised embedding generation using domain-agnostic sentence-transformers
        if "sentence-transformers" in self.model_name:
            last_hidden_state_representation = outputs.hidden_states[-1] # [batch_size, seq_len, hidden_dim]
            attention_mask = inputs["attention_mask"]
            sentence_embeddings = self.mean_pooling(last_hidden_state_representation, attention_mask)
        
        # Embedding generation using fine-tune RoBERTa model 
        else:
            # Pooling Strategy -> mean pooling (average of all token embeddings weighted by the attention mask for ignoring padded tokens)
            if not hasattr(self, "pooling_type") or self.pooling_type == "mean-pooling":
                last_hidden_state_representation = outputs.hidden_states[-1] 
                attention_mask = inputs["attention_mask"]
                sentence_embeddings = self.mean_pooling(last_hidden_state_representation, attention_mask)
            # Poolins Strategy -> cls token
            elif self.pooling_type == "cls-token":
                last_hidden_state_representation = outputs.hidden_states[-1]  
                sentence_embeddings = last_hidden_state_representation[:, 0, :] # cls token
            else:
                raise ValueError(f"Unsupported pooling type: {self.pooling_type}")
        
        return sentence_embeddings

    @staticmethod
    def mean_pooling(token_embeddings, attention_mask):
        """
        Performs mean pooling on token embeddings - taking the average of token representations weighted with the attention mask (paddings avoided with weight 0)
        Returns: Pooled embeddings of shape [batch_size, hidden_dim]
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9) # min defined so we avoid division by 0

# Example usage
if __name__ == "__main__":
    arguments = [
        "The player performed exceptionally well.",
        "The team needs better strategies to win.",
        "If Player X had scored, the outcome would be different."
    ]
    
    # Use Sentence-Transformers
    embedder_st = EmbeddingsGenerator(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embeddings_st = embedder_st.get_embeddings(arguments)
    print("Sentence-Transformer Embeddings:", embeddings_st.shape)

    # Use Fine-tuned RoBERTa model
    embedder_ft = EmbeddingsGenerator(model_name="../curr-pure-best")
    embeddings_ft = embedder_ft.get_embeddings(arguments)
    print("Fine-Tuned RoBERTa Embeddings:", embeddings_ft.shape)
