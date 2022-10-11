### for drawing plot in A2_Baseline model.

import torch
import os
from datetime import datetime
import matplotlib.pyplot as plt

def save_model(model):
    if not os.path.exists('./models'):
        os.mkdir('./models')
    timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")
    savedir = './models/model_cnn_{}.pt'.format(timestamp)
    return torch.save(model.state_dict(), savedir)
    
    
def save_results(results:list, args):
    timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")
    if not os.path.exists('./results'):
        os.mkdir('./results')
    savedir = './results/cnn_{}'.format(timestamp)
    if not os.path.exists('savedir'):
        os.mkdir(savedir)
    with open(savedir + '/result.txt', 'w') as f:
        f.write(str(vars(args)))
        f.write('\n')
        f.write('[train_loss, train_acc, val_loss, val_acc, test_acc]')
        for metric in results:
            f.write(f"{metric}\n")
            f.write('\n')
    draw_loss(results[0], results[2], savedir)
    draw_acc(results[1], results[3], savedir)
    return


def draw_loss(tloss, vloss, dir=None):
    if dir == None:
        savedir = './fig'
    else:
        savedir = dir + '/fig'
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(list(range(len(tloss))), tloss, color='r', label='Training')
    ax.plot(list(range(len(tloss)))[::5], vloss, color='b', label='Validation')
    ax.legend()
    ax.set_title('Train loss vs. Val loss')
    plt.savefig(savedir + '/loss.png')
    return fig


def draw_acc(tacc, vacc, dir=None):
    if dir == None:
        savedir = './fig'
    else:
        savedir = dir + '/fig'
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(list(range(len(tacc))), tacc, color='r', label='Training')
    ax.plot(list(range(len(tacc)))[::5], vacc, color='b', label='Validation')
    ax.legend()
    ax.set_title('Accuracy')
    plt.savefig(savedir + '/acc.png')
    return fig