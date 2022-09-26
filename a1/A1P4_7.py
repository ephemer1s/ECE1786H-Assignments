import numpy as np
import torch
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # For progress bars
import matplotlib.pyplot as plt
import os

from A1P4_4 import *
from A1P4_6 import SGNS

def train_sgns(textlist, w2i, window=5, embedding_size=8):
    # Set up a model with Skip-gram with negative sampling (predict context with word)
    # textlist: a list of strings
    np.random.seed(42)
    torch.manual_seed(42)
    # Create Training Data 
    X, T, Y = tokenize_and_preprocess_text(textlist, w2i, window)
    X, T, Y = np.array(X), np.array(T), np.array(Y)

    # Split the training data
    X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(X, T, Y, test_size=0.2, random_state=42)
    print(X_train.shape, X_test.shape, T_train.shape, T_test.shape, Y_train.shape, Y_test.shape)

    # instantiate the network & set up the optimizer

    model = SGNS(vocab_size=len(w2i.keys()), embedding_size=embedding_size)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    model.to(device)

    lr = 5e-4
    epochs = 30
    bs = 4
    n_workers = 1
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    # training loop
    centers = torch.from_numpy(X_train).split(bs)
    targets = torch.from_numpy(T_train).split(bs)
    labels = torch.from_numpy(Y_train).float().split(bs)

    progress_bar = tqdm(range(epochs * len(centers)))

    running_loss = []
    running_val_loss = []

    for epoch in range(epochs):
        epoch_loss = 0
        for center, target, label in zip(centers, targets, labels):
            center, target, label = center.to(device), target.to(device), label.to(device)
            optimizer.zero_grad()
            logits = model(x=center, t=target) # forward
            loss = loss_fn(logits, label)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            progress_bar.update(1)

        val_pred = model(x=torch.from_numpy(X_test), t=torch.from_numpy(T_test))
        val_loss = loss_fn(val_pred, torch.from_numpy(Y_test).float()).item()

        epoch_loss /= len(centers)
        running_loss.append(epoch_loss)
        running_val_loss.append(val_loss)

    return model, running_loss, running_val_loss


if __name__ == '__main__':
    with open('LargerCorpus.txt') as f:
        txt = f.read()
    filtered_lemmas, w2i, i2w, vocab = prepare_texts(txt)
    textlist = sent_tokenize(txt)
    model, tloss, vloss = train_sgns(textlist, w2i)
    
    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(tloss, 'r', label='Training')
    ax.plot(vloss, 'b', label='Validation')
    ax.legend()
    fig.savefig('./Report.assets/Section4-training-plot.png')
    
    if not os.path.exists('./models'):
        os.mkdir('./models')
    torch.save(model.state_dict(), './models/model')