import torch
import torchtext
# from torchtext import data
# import torch.optim as optim
import argparse
import os
from tqdm import tqdm
from datetime import datetime

try:
    from A2_Starter import *
    from Baseline import Baseline
except:
    from A2_Baseline.A2_Starter import *
    from A2_Baseline.Baseline import Baseline


# add args
parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch_size", type=int, default=16)
parser.add_argument("-e", "--epochs", type=int, default=50)
parser.add_argument("-l", "--learning_rate", type=float, default=1e-3)
args = parser.parse_args()


def main(args):
    # fix seed
    torch.manual_seed(2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print ("Using device:", device)

    ### 3.3 Processing of the data ###
    # 3.3.1
    # The first time you run this will download a 862MB size file to .vector_cache/glove.6B.zip
    glove = torchtext.vocab.GloVe(name="6B",dim=100) # embedding size = 100
                                   
    # 3.3.2
    train_dataset = TextDataset(glove, split="train")
    val_dataset = TextDataset(glove, split="validation")
    test_dataset = TextDataset(glove, split="test")
        
    # 3.3.3
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=lambda batch: my_collate_function(batch, device))

    validation_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=lambda batch: my_collate_function(batch, device))

    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: my_collate_function(batch, device))

    # Instantiate your model(s) and train them and so on 
    # We suggest parameterizing the model - k1, n1, k2, n2, and other hyperparameters
    # so that it is easier to experiment with

    ### 4.3 Training the Baseline Model ###
    # set up the model
    model = Baseline(glove).to(device)

    # set up hyperparameters
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer=torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)

    train_loss = []
    val_loss = []

    # training process
    progress_bar = tqdm(range(args.epochs))
    for epoch in range(args.epochs):

        epoch_loss = 0
        for X_train, Y_train in train_dataloader:  # train
            model.train()
            optimizer.zero_grad()
            out = model(X_train)
            loss = loss_fn(out, Y_train.float())
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        epoch_loss /= len(train_dataloader)
        train_loss.append(epoch_loss)

        epoch_loss = 0
        for X_val, Y_val in validation_dataloader:  # validation
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
            loss = loss_fn(val_pred, Y_val.float())#.item()
            epoch_loss += loss.item()
        epoch_loss /= len(validation_dataloader)
        val_loss.append(epoch_loss)

        progress_bar.update(1)
 
    # save model
    if not os.path.exists('./models'):
        os.mkdir('./models')
    timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")
    savedir = './models/Baseline_lr_{}_bs_{}_epochs_{}_{}'.format(
        args.learning_rate, args.batch_size, args.epochs, timestamp)
    torch.save(model.state_dict(), savedir)

    return model, train_loss, val_loss


if __name__ == '__main__':
    model, train_loss, val_loss = main(args)