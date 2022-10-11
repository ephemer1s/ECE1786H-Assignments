### extract meanings from trained parameters

import torch
import torchtext
from tqdm import tqdm

# try:
#     from A2_Starter import *
# except Exception as e: 
#     print(e)
#     print('trying another import path')
#     from A2_Baseline.A2_Starter import *
#     print('import successful')
    

glove = torchtext.vocab.GloVe(name="6B",dim=100) # embedding size = 100


def print_closest_cosine_words(vec:str, n:int):
    '''
    print out N-most similar word using cosine similarity
    '''
    dists = torch.cosine_similarity(glove.vectors, vec.unsqueeze(0))
    lst = sorted(enumerate(dists.numpy()), key=lambda x: x[1])
    out = []
    for idx, difference in lst[:-n-1:-1]:
        print(glove.itos[idx], "\t%5.2f" % difference)
        out.append([idx, difference])
    print('\n')

    return out   
        
if __name__ == "__main__":
    model = torch.load('A2_CNN/models/model_cnn_10112022_165525.pt')
    # print(model.state_dict())
    state = model.state_dict()
    kernel1 = state['conv1.0.weight'].squeeze().cpu().reshape(-1, 100)
    kernel2 = state['conv2.0.weight'].squeeze().cpu().reshape(-1, 100)
    kernels = torch.cat((kernel1, kernel2), 0)
    
    c_s = dict()
    # progress_bar = tqdm(range(kernels.shape[0]))
    
    for i in kernels:
        c_i = print_closest_cosine_words(i, 5)
        # for idx, diff in c_i:
        #     if glove.itos[idx] not in c_s:
        #         c_s[glove.itos[idx]] = diff
        #     else:
        #         c_s[glove.itos[idx]] += diff
        # progress_bar.update(1)
        
    
    # closest = [(k, v) for k, v in sorted(c_s.items(), key=lambda item: item[1])]
    # print(closest)