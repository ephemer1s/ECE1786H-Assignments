from collections import Counter
import numpy as np
import torch
from sklearn.model_selection import train_test_split

import spacy


def prepare_texts(text):    
    # prepare text using the spacy english pipeline (see https://spacy.io/models/en)
    # we'll use it to lemmatize the text, and determine which part of speech each
    # lemmatize edits words to become the 'root' word - e.g. holds -> hold;  rubs->rub
    # part of speech indicates if the item is a verb, nooun, punctuation, space and so on.
    # make sure that the text sent to spacy doesn't end with a period immediately followed by a newline,
    # instead, make sure there is a space between the period and the newline, so that the period 
    # is correctly identified as punctuation.
    # Get a callable object from spaCy that processes the text - lemmatizes and determines part of speech
    
    nlp = spacy.load("en_core_web_sm") # moved to import section
    
    # lemmatize the text, get part of speech, and remove spaces and punctuation
    
    lemmas = [tok.lemma_ for tok in nlp(text) if tok.pos_ not in ["PUNCT", "SPACE"]]
    
    # count the number of occurences of each word in the vocabulary
    
    freqs = Counter() 
    for w in lemmas:
        freqs[w] += 1
        
    vocab = sorted(list(freqs.items()), key=lambda item: item[1], reverse=True)  # List of (word, occurrence) Sort by decreasing frequency
    # print(vocab)
    
    # Create word->index dictionary and index->word dictionary
    
    v2i = {v[0]:i for i,v in enumerate(vocab)}
    i2v = {i:v[0] for i,v in enumerate(vocab)}
    
    return lemmas, v2i, i2v


def tokenize_and_preprocess_text(textlist, v2i, window=3):
    '''
    Predict context with word. Sample the context within a window size.
    X, Y = [], []  # is the list of training/test samples
    TO DO - create all the X,Y pairs'''
    samples = []
    n_grams = (window - 1) // 2
    for stc in textlist:
        lemma, _, _ = prepare_texts(stc)
        lemma = [v2i[i] for i in lemma] # transfer to indices
        for i, w in enumerate(lemma):
            for n in range(1, n_grams + 1):
                if i - n >= 0:
                    samples.append([w, lemma[i - n]])
                if i + n < len(lemma):
                    samples.append([w, lemma[i + n]])
    X, Y = train_test_split(samples, test_size=0.2, shuffle=True) # is the list of training/test samples
    return X, Y


    
    


if __name__ == "__main__":
    with open(r'SmallSimpleCorpus.txt') as f:
        text = f.read()
        
    lemmas, v2i, i2v = prepare_texts(text)
    # print(lemmas)  # used in Q3.2
    
    corpus = text.split('. ')  # separate sentences
    print(len(corpus))
    
    X, y = tokenize_and_preprocess_text(corpus[:10], v2i)
    print(X, y)