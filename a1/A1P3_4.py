import torch

class Word2vecModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.embedding = torch.nn.Embedding(self.vocab_size, self.embedding_size)
        self.expand = torch.nn.Linear(self.embedding_size, self.vocab_size, bias=False)

        # initialize word vectors to random numbers 
        init_weight = 0.5 / self.embedding_size
        self.embedding.weight.data.uniform_(-init_weight, init_weight)  # (-1, 1)
        
        # prediction function takes embedding as input, and predicts which word in vocabulary as output

        #TO DO
        
    def forward(self, x):
        """
        x: torch.tensor of shape (bsz), bsz is the batch size
        """

        # Encode input to lower-dimensional representation
        e = self.embedding(x)
        # Expand hidden layer to predictions
        logits = self.expand(e)
    
        return logits, e