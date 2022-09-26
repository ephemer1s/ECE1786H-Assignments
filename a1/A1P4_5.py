import torch

class SGNS(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.in_embed = torch.nn.Embedding(self.vocab_size, self.embedding_size)
        self.out_embed = torch.nn.Embedding(self.vocab_size, self.embedding_size)

        init_weight = 0.5 / self.embedding_size
        self.in_embed.weight.data.uniform_(-init_weight, init_weight)
        self.out_embed.weight.data.uniform_(-init_weight, init_weight)
        
    def forward(self, x, t):
        
        # x: torch.tensor of shape (batch_size), context word
        # t: torch.tensor of shape (batch_size), target ("output") word.
        center_embedding = self.in_embed(x)      # [batch_size, embedding_size]
        target_embedding = self.out_embed(t)     # [batch_size, embedding_size]
        
        logit = torch.bmm(target_embedding.unsqueeze(1), center_embedding.unsqueeze(2))
        logit = logit.squeeze().reshape([-1])
        logit = torch.sigmoid(logit)
        
        return logit