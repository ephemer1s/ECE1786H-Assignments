import torch 
import numpy as np

from nltk.tokenize import sent_tokenize 

from pathlib import Path 
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from mingpt.bpe import BPETokenizer 
from mingpt.utils import set_seed 
set_seed(1234)

print(f'cuda=={torch.cuda.is_available()}')

"""
Prepare the dataset to train the Language Model (LM)
This implementation splits the sentences and so doesn't create training 
examples that cross sentences.

This code is set so that it uses one of two possible datasets, which were also used in Assignment 1: 
SmallSimpleCorpus.txt or LargerCorpus.txt

Arguments:
            ds_choice: str. "small" or "large". (i.e. selects which of the two datasets)
            split: str. "train" or "test".
            truncation: int. If -1: no truncation on sentences. Otherwise: truncate to this specific length.
""" 

class LanguageModelingDataset(Dataset):
    
    def __init__(self, ds_choice="small", split="train", truncation=-1):
        
        base_path = "./"
        fn = {"small": "SmallSimpleCorpus.txt", "large": "LargerCorpus.txt"}
        self.ds_choice = ds_choice
        self.truncation = truncation  # int. If -1, then
        text = Path(base_path, fn[ds_choice]).read_text(encoding='utf-8')
        if ds_choice == "large":
            # Remove the newline char in the middle of sentences
            # The "paragraph splitting" newlines appear to be \n\n -- remove the duplications there
            text = text.replace("\n\n", "$$^^$$").replace("\n", " ").replace("$$^^$$", "\n")
        sentences = sent_tokenize(text)

        # Train / test split
        train, val = train_test_split(sentences, test_size=0.2, shuffle=False)
        if split == "train":
            raw_data = train 
        else:
            raw_data = val 

        # Tokenize
        self.tokenizer = BPETokenizer()
        self.data = []  # List of 1-d pytorch tensor
        for sent in raw_data:
            tokenized = self.tokenizer(sent).view(-1)  # pytorch tensor
            if truncation >= 0:
                self.data.append(tokenized[:truncation])
            else:
                self.data.append(tokenized)

        # Count some items
        self.max_sentence_length = np.max([len(d) for d in self.data])

    def __len__(self):
        return len(self.data)

    def get_vocab_size(self):
        """
        We have to set this to the max vocab size (i.e., that decided by the BPE tokenizer), 
        but actually, only a small number of vocab is used, especially for the small text. 
        """
        return 50257

    def __getitem__(self, idx):
        """
        The output should be a tuple x and y, both as pytorch tensors.
        Please refer to the `run()` method in the mingpt/trainer.py script for 
        how the x and y are going to be used.
        """
        x = self.data[idx][:-1]
        y = self.data[idx][1:]
        return (x, y)

    def get_block_size(self):
        """
        block_size is the size at which lines are truncated to ensure they are equal-length.
        """
        return self.max_sentence_length