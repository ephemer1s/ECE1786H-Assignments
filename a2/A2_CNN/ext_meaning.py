### extract meanings from trained parameters

import torch
import torchtext

try:
    from A2_Starter import *
except Exception as e: 
    print(e)
    print('trying another import path')
    from A2_Baseline.A2_Starter import *
    print('import successful')
    
# TODO: implemelent arg parse

glove = torchtext.vocab.GloVe(name="6B",dim=100) # embedding size = 100


def print_closest_cosine_words(vec:str, n:int):
    '''
    print out N-most similar word using cosine similarity
    '''
    dists = torch.cosine_similarity(glove.vectors, vec.unsqueeze(0))
    lst = sorted(enumerate(dists.numpy()), key=lambda x: x[1])
    for idx, difference in lst[:-n-1:-1]:
        print(glove.itos[idx], "\t%5.2f" % difference)
        
if __name__ == "__main__":
    model = torch.load('models/model_baseline_lr_0.001_bs_16_epochs_50_10082022_232310.pt')
    layer = model['linear.weight'].squeeze().cpu()
    print(layer.shape)
    print_closest_cosine_words(layer, 20)