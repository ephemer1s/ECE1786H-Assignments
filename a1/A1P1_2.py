import torch
import torchtext
import numpy

import warnings
warnings.filterwarnings("ignore")

glove = torchtext.vocab.GloVe(name="6B", dim=50)

def print_closest_words(vec, n):
    '''
    the original function borrowed from A1_Section1_starter.ipynb
    print out N-most similar word using ED
    '''
    dists = torch.norm(glove.vectors - vec, dim=1)     # compute distances to all words
    lst = sorted(enumerate(dists.numpy()), key=lambda x: x[1]) # sort by distance
    
    for idx, difference in lst[1:n+1]:
        print(glove.itos[idx], "\t%5.2f" % difference)
        
        
def print_closest_cosine_words(vec:str, n:int):
    '''
    print out N-most similar word using cosine similarity
    '''
    dists = torch.cosine_similarity(glove.vectors, vec.unsqueeze(0))
    lst = sorted(enumerate(dists.numpy()), key=lambda x: x[1])
    for idx, difference in lst[:-n-1:-1]:
        print(glove.itos[idx], "\t%5.2f" % difference)
        

if __name__ == "__main__":
    print_closest_cosine_words(glove['dog'], 10)
    print_closest_words(glove['dog'], 10)
    
    print_closest_cosine_words(glove['computer'], 10)
    print_closest_words(glove['computer'], 10)