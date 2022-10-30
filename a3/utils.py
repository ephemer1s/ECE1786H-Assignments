import torch
import matplotlib.pyplot as plt
import os
import numpy as np

def lm_collate_fn(batch, device):
    x = [item[0] for item in batch]  # List (len B) of varying lengths
    y = [item[1] for item in batch]  # List (len B) of the same lengths as x
    maxlen = max([len(s) for s in x])

    padded_x, padded_y = [], []
    for sx, sy in zip(x, y):
        padded_x.append(torch.cat([sx, torch.ones(maxlen - len(sx))]))
        padded_y.append(torch.cat([sy, torch.ones(maxlen - len(sy))]))
    return torch.stack(padded_x).long().to(device), torch.stack(padded_y).long().to(device)


def batch_end_callback(trainer):
    '''
    # This function is called at the end of every batch in training
    # and is used to report the amount of time per 100 batches, and the loss at that point
    '''
    if trainer.iter_num % 100 == 0:
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f} acc {trainer.iter_acc:.3f};")
        

def sen_collate_fn(batch, device):
    x = [item[0] for item in batch]  # List (len B) of varying lengths
    y = [item[1] for item in batch]  # List (len B) of the same lengths as x
    # print(x, y)
    maxlen = max([len(s) for s in x])

    padded_x = []
    for sx in x:
        padded_x.append(torch.cat([sx, torch.ones(maxlen - len(sx))]))
        
    ret_x = torch.stack(padded_x).long().to(device)
    ret_y = torch.as_tensor(y).long().to(device)
    return ret_x, ret_y

def sen_batch_end_callback(trainer):
    '''
    # This function is called at the end of every batch in training
    # and is used to report the amount of time per 100 batches, and the loss at that point
    '''
    if trainer.iter_num % 100 == 0:
        print(f"iter{trainer.iter_num} {trainer.iter_dt * 1000:.2f}ms; train loss {trainer.loss.item():.5f} acc {trainer.acc:.3f}; val loss {trainer.vloss.item():.5f} acc {trainer.vacc:.3f};")
        
        
def plot_loss(trainer, dir=None):
    if dir == None:
        savedir = './fig'
    else:
        savedir = dir + '/fig'
    if not os.path.exists(savedir):
        os.mkdir(savedir)
        
    window = 20
    
    average_tloss = []
    for ind in range(len(trainer.train_loss) - window + 1):
        average_tloss.append(np.mean(trainer.train_loss[ind:ind+window]))
    for ind in range(window - 1):
        average_tloss.insert(0, np.nan)
        
    average_vloss = []
    for ind in range(len(trainer.val_loss) - window + 1):
        average_vloss.append(np.mean(trainer.val_loss[ind:ind+window]))
    for ind in range(window - 1):
        average_vloss.insert(0, np.nan)
        
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    axs[0].plot(trainer.train_loss, alpha=0.5)
    axs[0].plot(average_tloss, alpha=1)
    axs[1].plot(trainer.val_loss, alpha=0.5)
    axs[1].plot(average_vloss, alpha=1)
    
    axs[0].set_title('Training Loss')
    axs[1].set_title('Validation Loss')
    plt.savefig(savedir + '/loss.png')
    return fig


def plot_acc(trainer, dir=None):
    if dir == None:
        savedir = './fig'
    else:
        savedir = dir + '/fig'
    if not os.path.exists(savedir):
        os.mkdir(savedir)
        
    window = 20
    
    average_tacc = []
    for ind in range(len(trainer.train_acc) - window + 1):
        average_tacc.append(np.mean(trainer.train_acc[ind:ind+window]))
    for ind in range(window - 1):
        average_tacc.insert(0, np.nan)
        
    average_vacc = []
    for ind in range(len(trainer.val_acc) - window + 1):
        average_vacc.append(np.mean(trainer.val_acc[ind:ind+window]))
    for ind in range(window - 1):
        average_vacc.insert(0, np.nan)
    
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    axs[0].plot(trainer.train_acc, alpha=0.5)
    axs[0].plot(average_tacc, alpha=1)
    axs[1].plot(trainer.val_acc, alpha=0.5)
    axs[1].plot(average_vacc, alpha=1)
    
    axs[0].set_title('Training Accuracy')
    axs[1].set_title('Validation Accuracy')
    plt.savefig(savedir + '/acc.png')
    return fig
