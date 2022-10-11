### for drawing plot in A2_Baseline model.


import matplotlib.pyplot as plt


def draw_loss(tloss, vloss):
    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(list(range(len(tloss))), tloss, color='r', label='Training')
    ax.plot(list(range(len(tloss)))[::2], vloss, color='b', label='Validation')
    ax.legend()
    ax.set_title('Train loss vs. Val loss')
    plt.savefig('.A2_Baseline/fig/loss.png')
    return fig

def draw_acc(acc):
    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(list(range(len(acc) * 2))[::2], acc, color='b')
    ax.set_title('Validation Accuracy')
    plt.savefig('.A2_Baseline/fig/acc.png')
    return fig