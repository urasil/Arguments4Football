from transformers import AutoModel, AutoTokenizer, RobertaForSequenceClassification, RobertaTokenizer
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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

        if "sentence-transformers" in self.model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        else:  
            self.tokenizer = RobertaTokenizer.from_pretrained("../curr-pure-best")
            self.model = RobertaForSequenceClassification.from_pretrained("../curr-pure-best").to(self.device)

    def get_embeddings(self, arguments):
        """
        Compute embeddings for a list of arguments
        Returns: List[List[int]] - List[Embeddings]
        """

        inputs = self.tokenizer(arguments, padding=True, truncation=True, return_tensors="pt").to(self.device)
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




    def visualize_embeddings(self, embeddings):
        """
        Visualize embeddings using PCA for dimensionality reduction to 2D.
        """
        embeddings_np = embeddings.cpu().numpy() if hasattr(embeddings, 'cpu') else embeddings
        reduced_embeddings = PCA(n_components=2).fit_transform(embeddings_np)

        plt.figure(figsize=(10, 8))
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c='blue', alpha=0.7)

        num_points = len(reduced_embeddings)
        for i in range(num_points):
            plt.annotate(
                str(i),
                (reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
                fontsize=8,
                alpha=0.75,
            )

        plt.title("Embedding Space (PCA)")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.grid(alpha=0.3)
        plt.show()

# Example usage
if __name__ == "__main__":
    arguments = [
        "Player X's goal in the final minute proves his incredible composure under pressure. It was the decisive moment of the match.",
        "Team Y's defensive strategy failed because they left gaps in the midfield. This allowed Player Z to exploit the space and score.",
        "The referee made a controversial decision, awarding a penalty to Team A that changed the outcome of the game.",
        "Player B's performance was outstanding as he scored a hat-trick, carrying his team to victory.",
        "If the weather conditions had been better, Team C might have performed more effectively in the match.",
        "Team D dominated possession but failed to convert their chances, highlighting their inefficiency in front of the goal."
    ]
    
    # Use Sentence-Transformers
    embedder_st = EmbeddingsGenerator(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embeddings_st = embedder_st.get_embeddings(arguments)
    print("Sentence-Transformer Embeddings:", embeddings_st.shape)

    embedder_st.visualize_embeddings(embeddings=embeddings_st)

    # Use Fine-tuned RoBERTa model
    embedder_ft = EmbeddingsGenerator(model_name="../curr-pure-best")
    embeddings_ft = embedder_ft.get_embeddings(arguments)
    print("Fine-Tuned RoBERTa Embeddings:", embeddings_ft.shape)

    embedder_ft.visualize_embeddings(embeddings=embeddings_ft)