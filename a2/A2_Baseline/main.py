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
    from plot import *
except Exception as e: 
    print(e)
    print('trying another import path')
    from A2_Baseline.A2_Starter import *
    from A2_Baseline.Baseline import Baseline
    from A2_Baseline.plot import *
    print('import successful')

# 4.3 parse args
parser = argparse.ArgumentParser()
parser.add_argument("-bs", "--batch_size", type=int, default=16)
parser.add_argument("-e", "--epochs", type=int, default=50)
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)
parser.add_argument("-ml", "--max_len", type=int, default=0)
parser.add_argument("-s", "--save_model", type=bool, default=True)
parser.add_argument("-o", "--overfit_debug", type=bool, default=False)
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
    if args.overfit_debug:
        train_dataset = TextDataset(glove, split="overfit")
    else:
        train_dataset = TextDataset(glove, split="train")
    val_dataset = TextDataset(glove, split="validation")
    test_dataset = TextDataset(glove, split="test")
        
    # 3.3.3
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=lambda batch: my_collate_function(batch, device, max_len=args.max_len))

    validation_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=lambda batch: my_collate_function(batch, device, max_len=args.max_len))

    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: my_collate_function(batch, device, max_len=args.max_len))

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
    val_acc = []

    # 4.3 training loop
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
        train_loss.append(epoch_loss)  # sum up train loss


        # do validation per 2 epochs for speed up
        if epoch % 2 == 0:
            epoch_loss = 0
            acc = 0
            for X_val, Y_val in validation_dataloader:  # validation
                model.eval()
                with torch.no_grad():
                    val_pred = model(X_val)
                loss = loss_fn(val_pred, Y_val.float())
                epoch_loss += loss.item()
                
                logit = torch.sigmoid(val_pred)
                Y_pred = torch.round(logit).long()
                # print(Y_pred, Y_val)
                for i in range(args.batch_size):
                    if Y_pred[i] == Y_val[i]:
                        acc += 1
            
            epoch_loss /= len(validation_dataloader)    # len = 1600 / bs
            val_loss.append(epoch_loss)                 # sum up val loss
            
            acc /= len(val_dataset)                     # len = 1600
            val_acc.append(acc)                         # sum up val acc
        
        # finish epoch
        progress_bar.update(1)
 
    # 4.5 Evaluation loop
    model.eval()
    test_acc = 0
    for X_test, Y_test in test_dataloader:
        with torch.no_grad():
            out = model(X_test)
        logit = torch.sigmoid(out)
        Y_pred = torch.round(logit).long()
        for i in range(args.batch_size):
            if Y_pred[i] == Y_test[i]:
                test_acc += 1
    test_acc /= len(test_dataset)
    print("test accuracy: {}".format(test_acc))
    
    # 4.7 save model
    if args.save_model:
        if not os.path.exists('./models'):
            os.mkdir('./models')
        timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")
        savedir = './models/model_baseline_lr_{}_bs_{}_epochs_{}_{}.pt'.format(
            args.learning_rate, args.batch_size, args.epochs, timestamp)
        torch.save(model.state_dict(), savedir)

    # 4.5 Draw curves
    if not os.path.exists('./fig'):
        os.mkdir('./fig')
    draw_loss(train_loss, val_loss)
    draw_acc(val_acc)
    
    # finally, return model and losses
    return model, train_loss, val_loss, val_acc, test_acc
    
    
    
if __name__ == '__main__':
    model, train_loss, val_loss, val_acc, test_acc = main(args)
    
