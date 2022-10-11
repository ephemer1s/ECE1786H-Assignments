import torch

class Baseline(torch.nn.Module):
    def __init__(self, vocab, fc_bias=False):
        super().__init__()

        self.embedding_size = vocab.vectors.shape[1]  # 100
        self.embedding = torch.nn.Embedding.from_pretrained(vocab.vectors)
        self.linear = torch.nn.Linear(self.embedding_size, 1, bias=fc_bias)

        # init_weight = 0.5 / self.embedding_size
        # self.embedding.weight.data.uniform_(-init_weight, init_weight)
        return

    def forward(self, x):                   # (max_len, bs)
        emb = self.embedding(x)             # (max_len, bs, esize)
        mean = torch.mean(emb, 0, True)     # (1, bs, esize)
        out = self.linear(mean)             # (1, bs, 1)
        return out.squeeze()                # (bs)