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
    
    
def save_results(results:list):
    if not os.path.exists('./results'):
        os.mkdir('./results')
    timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")
    savedir = './results/cnn_{}.pt'.format(timestamp)
    # TODO: save results
    pass
    
    
def save_args(args):
    # TODO: save args
    pass


def draw_loss(tloss, vloss):
    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(list(range(len(tloss))), tloss, color='r', label='Training')
    ax.plot(list(range(len(tloss)))[::2], vloss, color='b', label='Validation')
    ax.legend()
    ax.set_title('Train loss vs. Val loss')
    plt.savefig('./fig/loss.png')
    return fig


def draw_acc(acc):
    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(list(range(len(acc) * 2))[::2], acc, color='b')
    ax.set_title('Validation Accuracy')
    plt.savefig('./fig/acc.png')
    return fig