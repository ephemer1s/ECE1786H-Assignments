import math
import torch
import torchtext
import argparse
# import os
from tqdm import tqdm
from dotmap import DotMap
import json

try:
    from A2_Starter import *
    from Classifier import Conv2dWordClassifier
    from utils import *
except Exception as e: 
    print(e)
    print('trying another import path')
    from A2_CNN.A2_Starter import *
    from A2_CNN.Classifier import Conv2dWordClassifier
    from A2_CNN.utils import *
    print('import successful')

# 4.3 parse args
parser = argparse.ArgumentParser()
parser.add_argument("-bs", "--batch_size", type=int, default=64)
parser.add_argument("-e", "--epochs", type=int, default=50)
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)
parser.add_argument("-ml", "--max_len", type=int, default=0)
parser.add_argument("-s", "--save_model", type=bool, default=True)
parser.add_argument("-o", "--overfit_debug", type=bool, default=False)
parser.add_argument("-b", "--bias", type=bool, default=False)

# added in 5.2
parser.add_argument("-k1", "--k1", type=int, default=2)
parser.add_argument("-n1", "--n1", type=int, default=10)
parser.add_argument("-k2", "--k2", type=int, default=4)
parser.add_argument("-n2", "--n2", type=int, default=10)
parser.add_argument("-f", "--freeze_embedding", type=bool, default=False)
parser.add_argument("-g", "--grid_search", type=bool, default=False)

# 3.3.1
# The first time you run this will download a 862MB size file to .vector_cache/glove.6B.zip
glove = torchtext.vocab.GloVe(name="6B",dim=100) # embedding size = 100

def preprocess(args):
    # fix seed
    torch.manual_seed(2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print ("Using device:", device)

    ### 3.3 Processing of the data ###
                                   
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

    return train_dataloader, validation_dataloader, test_dataloader


def main(args, train_dataloader, validation_dataloader, test_dataloader):
    '''
    "3.3 Processing of the data" was moved to function processing() above
    '''
   
    # Instantiate your model(s) and train them and so on 
    # We suggest parameterizing the model - k1, n1, k2, n2, and other hyperparameters
    # so that it is easier to experiment with

    ### 4.3 Training the Baseline Model ###
    # set up the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = Conv2dWordClassifier(glove, args).to(device)

    # set up hyperparameters
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer=torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)
    
    # set up progress bar
    epochs_per_val = 5     # do validation per this epochs
    
    num_iters = args.epochs * len(train_dataloader) \
        + args.epochs // epochs_per_val * len(validation_dataloader)
    progress_bar = tqdm(range(int(num_iters)))
    
    # 5.2 training loop, transplanted from 4.3 training loop
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    
    for epoch in range(args.epochs):

        epoch_loss = 0
        acc = 0
        for X_train, Y_train in train_dataloader:  # train
            model.train()
            optimizer.zero_grad()
            out, logit = model(X_train)
            loss = loss_fn(out, Y_train.float())
            epoch_loss += loss.item()

            Y_pred = torch.round(logit).long()
            # print(Y_pred.shape, Y_train.shape)
            for i in range(min(args.batch_size, len(Y_pred))):
                if Y_pred[i] == Y_train[i]:
                    acc += 1
                        
            loss.backward()
            optimizer.step()
            progress_bar.update(1)
            
        epoch_loss /= len(train_dataloader)
        train_loss.append(epoch_loss)  # sum up train loss
        
        acc /= len(train_dataloader.dataset)                   # len = 6400
        train_acc.append(acc)                       # sum up val acc

        # do validation per 5 epochs for speed up
        if epoch % 5 == 0:
            epoch_loss = 0
            acc = 0
            for X_val, Y_val in validation_dataloader:  # validation
                model.eval()
                with torch.no_grad():
                    out, logit = model(X_val)
                loss = loss_fn(out, Y_val.float())
                epoch_loss += loss.item()
                
                Y_pred = torch.round(logit).long()
                for i in range(min(args.batch_size, len(Y_pred))):
                    if Y_pred[i] == Y_val[i]:
                        acc += 1
                        
                progress_bar.update(1)
                
            epoch_loss /= len(validation_dataloader)    # len = 1600 / bs
            val_loss.append(epoch_loss)                 # sum up val loss
            
            acc /= len(validation_dataloader.dataset)                     # len = 1600
            val_acc.append(acc)                         # sum up val acc
        
    # finish epoch
 
    # 4.5 Evaluation loop
    model.eval()
    test_acc = 0
    for X_test, Y_test in test_dataloader:
        with torch.no_grad():
            out, logit = model(X_test)
        Y_pred = torch.round(logit).long()
        for i in range(min(args.batch_size, len(Y_pred))):
            if Y_pred[i] == Y_test[i]:
                test_acc += 1
    test_acc /= len(test_dataloader.dataset)
    print("test accuracy: {}".format(test_acc))
    

    # 5.2.1 
    # finally, return model and losses
    return model, train_loss, train_acc, val_loss, val_acc, test_acc
    
    
def grid_search():
    '''
    Using Grid Search to tune the CNN model.
    Using args from gs_args.json. Parser args are blocked
    '''
    best_acc = 0
    best_result = []
    best_args = []
    
    with open('A2_CNN/gs_args.json', 'r') as f:
        data = json.load(f)
    d_args = DotMap(data["default_args"])
    train_dataloader, validation_dataloader, test_dataloader = preprocess(d_args)
    
    for lr in data["grid_args"]["learning_rate"]:
        for k1 in data["grid_args"]["k1"]:
            for k2 in data["grid_args"]["k2"]:
                if k1 <= k2:
                    for n1 in data["grid_args"]["n1"]:
                        for n2 in data["grid_args"]["n2"]:
                            gs_args = dict()

                            gs_args["learning_rate"] = lr
                            gs_args["k1"] = k1
                            gs_args["k2"] = k2
                            gs_args["n1"] = n1
                            gs_args["n2"] = n2
                            
                            print(gs_args)
                            
                            for i in data["default_args"].keys():
                                gs_args[i] = data["default_args"][i]

                            args = DotMap(gs_args)
                            
                            model, train_loss, train_acc, val_loss, val_acc, test_acc = main(args, train_dataloader, validation_dataloader, test_dataloader)
                            
                            # print("test_acc == {}" + str(test_acc))
                            
                            if test_acc > best_acc:
                                best_acc = test_acc
                                best_model = model
                                best_result = [train_loss, train_acc, val_loss, val_acc, test_acc]
                                best_args = gs_args

                        
    return best_model, best_result, best_args
    
if __name__ == '__main__':
    args = parser.parse_args()
    if args.grid_search:
        best_model, best_result, best_args = grid_search()
        
        save_results(best_result, best_args)
        
        
    else:
        train_dataloader, validation_dataloader, test_dataloader = preprocess(args)
        model, train_loss, train_acc, val_loss, val_acc, test_acc = main(args, train_dataloader, validation_dataloader, test_dataloader)
    
        # if args.save_model:
            # 4.7 save model 
            # save_model(model)

            # 4.5 Draw curves
            # draw_loss(train_loss, val_loss)
            # draw_acc(train_acc, val_acc)
            
        # save results
        save_results([train_loss, train_acc, val_loss, val_acc, test_acc], args)
    
