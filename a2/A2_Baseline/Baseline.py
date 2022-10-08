import torch

class Baseline(torch.nn.Module):
    def __init__(self, vocab, fc_bias=False):
        super().__init__()

        self.embedding_size = vocab.vectors.shape[1]  # 100

        self.embedding = torch.nn.Embedding.from_pretrained(vocab.vectors)       # (batch, 1, len(x)) (batch, 100, len(x))
        self.linear = torch.nn.Linear(self.embedding_size, 1, bias=fc_bias)      # (batch, 100, 1) (batch, 1, 1)

        # init_weight = 0.5 / self.embedding_size
        # self.embedding.weight.data.uniform_(-init_weight, init_weight)
        return

    def forward(self, x):
        return self.linear(torch.mean(self.embedding(x)))