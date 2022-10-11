### for drawing plot in A2_Baseline model.

import torch
import os
from datetime import datetime
import matplotlib.pyplot as plt

def save_model(model):
    if not os.path.exists('./A2_CNN/models'):
        os.mkdir('./A2_CNN/models')
    timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")
    savedir = './A2_CNN/models/model_cnn_{}.pt'.format(timestamp)
    return torch.save(model.state_dict(), savedir)
    
    
def save_results(results:list, args):
    timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")
    if not os.path.exists('./A2_CNN/results'):
        os.mkdir('./A2_CNN/results')
    savedir = './A2_CNN/results/cnn_{}'.format(timestamp)
    if not os.path.exists('savedir'):
        os.mkdir(savedir)
    with open(savedir + '/result.txt', 'w') as f:
        f.write(str(vars(args)))
        f.write('\n')
        f.write('[train_loss, train_acc, val_loss, val_acc, test_acc]')
        for metric in results:
            f.write(f"{metric}\n")
            f.write('\n')
    draw_loss(results[0], results[2], args.val_epoch, savedir)
    draw_acc(results[1], results[3], args.val_epoch, savedir)
    return


def draw_loss(tloss, vloss, epv, dir=None):
    if dir == None:
        savedir = './fig'
    else:
        savedir = dir + '/fig'
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(list(range(len(tloss))), tloss, color='r', label='Training')
    ax.plot(list(range(len(tloss)))[::epv], vloss, color='b', label='Validation')
    ax.legend()
    ax.set_title('Train loss vs. Val loss')
    plt.savefig(savedir + '/loss.png')
    return fig


def draw_acc(tacc, vacc, epv, dir=None):
    if dir == None:
        savedir = './fig'
    else:
        savedir = dir + '/fig'
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(list(range(len(tacc))), tacc, color='r', label='Training')
    ax.plot(list(range(len(tacc)))[::epv], vacc, color='b', label='Validation')
    ax.legend()
    ax.set_title('Accuracy')
    plt.savefig(savedir + '/acc.png')
    return fig