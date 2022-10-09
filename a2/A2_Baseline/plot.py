### for drawing plot in A2_Baseline model.


import matplotlib.pyplot as plt


def draw_loss(tloss, vloss):
    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(x=list(range(len(tloss))), y=tloss, color='r', label='Training')
    ax.plot(x=list(range(len(tloss)))[::2], y=vloss, color='b', label='Validation')
    ax.legend()
    ax.set_title('Train loss vs. Val loss')
    ax.show()
    return fig

def draw_acc(acc):
    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(x=list(range(len(acc) * 2))[::2], y=acc, color='b', title='Validation Accuracy')