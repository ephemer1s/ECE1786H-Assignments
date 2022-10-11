import torch
from torch import nn


class Conv2dWordClassifier(nn.Module):
    def __init__(self, vocab, args):
        super().__init__()

        self.k1 = (args.k1, 100)                        # rectan kernel size (k * 100), parallel to (N * d)
        self.k2 = (args.k2, 100)                        # k1 and k2 are tuples
        self.n1 = args.n1
        self.n2 = args.n2

        self.linear_input_size = self.n1 + self.n2      # as per 5.0.6

        ### set up neural network
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors, freeze=args.freeze_embedding)  # same as Baseline

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=self.n1, bias=False,    # input: (bs, ch=1, height=N, width=100)
                kernel_size=self.k1),                               # output: (bs, ch=n1, height=N-k1+1, width=1)
            # nn.BatchNorm2d(num_features=self.n1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(output_size=(1, 1)),               # kernel_size = input_size
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=self.n2, bias=False,    # input: (bs, ch=1, height=N, width=100)
                kernel_size=self.k2),                               # output: (bs, ch=n2, height=N-k2+1, width=1)
            # nn.BatchNorm2d(num_features=self.n2),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(output_size=(1, 1)),               # kernel_size = input_size
        )

        self.linear = nn.Linear(self.linear_input_size, 1, bias=args.bias)
        self.sigmoid = nn.Sigmoid()

        return


    def forward(self, x):                                       # (N, bs)
        emb = self.embedding(x)                                 # (N, bs, 100)
        conv_input = torch.transpose(emb, 0, 1).unsqueeze(1)    # (bs, 1, N, 100)
        conv_out1 = self.conv1(conv_input)                      # (bs, n1, 1, 1)
        conv_out2 = self.conv2(conv_input)                      # (bs, n2, 1, 1)
        cat = torch.cat((conv_out1, conv_out2), dim=1)          # (bs, n1+n2, 1, 1)
        out = self.linear(cat.squeeze())                        # (bs, 1)
        logit = self.sigmoid(out)
        
        return out.squeeze(), logit.squeeze()
