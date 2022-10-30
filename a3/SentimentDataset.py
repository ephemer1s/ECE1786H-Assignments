import torch 
import numpy as np

from nltk.tokenize import sent_tokenize 

from pathlib import Path 
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from mingpt.bpe import BPETokenizer 
from mingpt.utils import set_seed 
set_seed(1234)

from mingpt.model import GPT
from mingpt.trainer import Trainer
from utils import *
import datasets


class SentimentDataset(Dataset):
    
    # def __init__(self, ds_choice="small", split="train", truncation=-1):
    def __init__(self, split="train", truncation=-1):
        
        self.truncation = truncation  # int. If -1, then

        sst = datasets.load_dataset('glue', 'sst2')
        raw_data = sst[split]
        self.tokenizer = BPETokenizer()
        self.data = []  # List of 1-d pytorch tensor
        for data in raw_data:
            tokenized = self.tokenizer(data['sentence']).view(-1)  # pytorch tensor
            if truncation >= 0:
                self.data.append((tokenized[:truncation], data['label']))
            else:
                self.data.append((tokenized, data['label']))
        
        
        # sst = datasets.load_dataset('glue', 'sst2')['train'][:1200]
        # tx, vx, ty, vy = train_test_split(sst['sentence'], sst['label'], test_size=0.2, shuffle=False)
        # if split == "train":
        #     raw_data, label = tx, ty
        # else:
        #     raw_data, label = vx, vy
        # self.tokenizer = BPETokenizer()
        # self.data = []  # List of 1-d pytorch tensor
        # for (x, y) in zip(raw_data, label):
        #     tokenized = self.tokenizer(x).view(-1)  # pytorch tensor
        #     self.data.append((tokenized, y))

        self.max_sentence_length = 512

    def __len__(self):
        return len(self.data)

    def get_vocab_size(self):
        """
        We have to set this to the max vocab size (i.e., that decided by the BPE tokenizer), 
        but actually, only a small number of vocab is used, especially for the small text. 
        """
        encoder = self.tokenizer.encoder.encoder
        return len(encoder.items())

    def __getitem__(self, idx):
        """
        The output should be a tuple x and y, both as pytorch tensors.
        Please refer to the `run()` method in the mingpt/trainer.py script for 
        how the x and y are going to be used.
        """
        x = self.data[idx][0]
        y = self.data[idx][-1]
        return (x, y)

    def get_block_size(self):
        """
        block_size is the size at which lines are truncated to ensure they are equal-length.
        """
        return self.max_sentence_length