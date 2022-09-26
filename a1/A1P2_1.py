import torch
import torchtext

glove = torchtext.vocab.GloVe(name="6B", # trained on Wikipedia 2014 corpus
                              dim=50)    # embedding size = 50

import numpy as np


def compare_words_to_category(cat, vec):
    sims = []
    for tar in cat:
        cossim = torch.cosine_similarity(tar, vec.unsqueeze(0))
        sims.append(cossim)
    average1 = sum(sims) / len(sims)
    mean_embedding = torch.mean(torch.cat(cat, dim=0), 0)
    average2 = torch.cosine_similarity(mean_embedding.unsqueeze(0), vec.unsqueeze(0))
    return average1, average2


cat = [
    glove['hot'].unsqueeze(0), 
    glove['warm'].unsqueeze(0), 
    glove['cool'].unsqueeze(0), 
    glove['cold'].unsqueeze(0), 
    glove['freeze'].unsqueeze(0)
]
vec = glove['apple']
a1, a2 = compare_words_to_category(cat, vec)
print(a1, a2)