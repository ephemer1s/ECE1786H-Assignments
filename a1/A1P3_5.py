import numpy as np
import spacy
from collections import Counter
import torch
from sklearn.model_selection import train_test_split
from A1P3_4 import Word2vecModel
from tqdm import tqdm  # For progress bars
import matplotlib.pyplot as plt

v2i = {
    'and': 0,
    'hold': 1,
    'dog': 2,
    'cat': 3,
    'rub': 4,
    'a': 5,
    'the': 6,
    'can': 7,
    'she': 8,
    'he': 9,
    'I': 10
}


def prepare_texts(text):    
    # Get a callable object from spaCy that processes the text - lemmatizes and determines part of speech

    nlp = spacy.load("en_core_web_sm")
    
    # lemmatize the text, get part of speech, and remove spaces and punctuation
    
    lemmas = [tok.lemma_ for tok in nlp(text) if tok.pos_ not in ["PUNCT", "SPACE"]]
    
    # count the number of occurences of each word in the vocabulary
    
    freqs = Counter() 
    for w in lemmas:
        freqs[w] += 1
        
    vocab = list(freqs.items())  # List of (word, occurrence)
    
    vocab = sorted(vocab, key=lambda item: item[1], reverse=True)  # Sort by decreasing frequency
    # print(vocab)
    
    # Create word->index dictionary and index->word dictionary
    
    v2i = {v[0]:i for i,v in enumerate(vocab)}
    i2v = {i:v[0] for i,v in enumerate(vocab)}
    
    return lemmas, v2i, i2v


def tokenize_and_preprocess_text(textlist, v2i, window=5):
    X, Y = [], []
    n_grams = (window - 1) // 2

    for stc in textlist:
        lemma, _, _ = prepare_texts(stc)
        lemma = [v2i[i] for i in lemma] # transfer to indices

        for i, w in enumerate(lemma):
            for n in range(1, n_grams + 1):
                if i - n >= 0:
                    X.append(w)
                    Y.append(lemma[i - n])
                if i + n < len(lemma):
                    X.append(w)
                    Y.append(lemma[i + n])

    return np.array(X, dtype=int), np.array(Y, dtype=int)


def train_word2vec(textlist, window=5, embedding_size=2):
    '''
    Set up a model with Skip-gram (predict context with word)
    textlist: a list of the strings
    '''
    # Create the training data
    X, y = tokenize_and_preprocess_text(textlist, v2i) # moved to front for speed
    print (X.shape, y.shape)

    # Split the training data
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # instantiate the network & set up the optimizer
    
    model = Word2vecModel(vocab_size=len(v2i.keys()), embedding_size=embedding_size)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    lr = 1e-3
    epochs = 50
    bs = 4
    n_workers = 1
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    # training loop
    batches = torch.from_numpy(X_train).split(bs)
    targets = torch.from_numpy(y_train).split(bs)

    progress_bar = tqdm(range(epochs))

    running_loss = []
    running_val_loss = []

    for epoch in range(epochs):
        epoch_loss = 0
        for center, context in zip(batches, targets):
            center, context = center.to(device), context.to(device)
            optimizer.zero_grad()
            logits, e = model(x=center) # forward
            loss = loss_fn(logits, context)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        val_pred, _ = model(x=torch.from_numpy(X_test))
        val_loss = loss_fn(val_pred, y_test).item()

        progress_bar.update(1)
        epoch_loss /= len(batches)
        running_loss.append(epoch_loss)
        running_val_loss.append(val_loss)

    return model, running_loss, running_val_loss


if __name__ == '__main__':
    with open('SmallSimpleCorpus.txt') as f:
        corpus = f.readline()
    corpus = corpus.split('. ')  # separate sentences
    network, tloss, vloss = train_word2vec(corpus)
    embedding = network.embedding
    
    fig = plt.figure()
    ax = fig.subplots(1, 1, 1)
    ax.plot(tloss, 'r')
    ax.plot(vloss, 'b')
    
    
    