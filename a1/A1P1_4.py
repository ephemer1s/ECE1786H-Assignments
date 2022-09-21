import torch
import torchtext

import warnings
warnings.filterwarnings("ignore")


glove = torchtext.vocab.GloVe(name="6B", dim=50)

def print_city_nation(city:str, n=1):
    '''generate the second word given the first word 
    the word-pair relationships given in0 Table 1 of the Mikolov paper'''
    vec = glove[city] - glove['city'] + glove['nation']
    dists = torch.cosine_similarity(glove.vectors, vec.unsqueeze(0))
    lst = sorted(enumerate(dists.numpy()), key=lambda x: x[1])
    for idx, difference in lst[:-n-1:-1]:
        print(glove.itos[idx], "\t%5.2f" % difference)
        

if __name__ == '__main__':
    cities = [
        'beijing', 'tokyo', 'seoul', 'london', 'paris', 'athens', 'ottawa', 'leningrad', 'washington', 'riyadh'
    ]
    for i in cities:
        print(i)
        print_city_nation(i, n=5)
        print('\n')