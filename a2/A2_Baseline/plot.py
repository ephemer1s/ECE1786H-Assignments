### for drawing plot in A2_Baseline model.


import matplotlib.pyplot as plt


def draw_loss(tloss, vloss):
    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(x=list(range(args.epochs)), y=train_loss, color='r', label='Training')
    ax.plot(x=list(range(args.epochs * 2))[::2], y=val_loss, color='b', label='Validation')
    ax.legend()
    ax.set_title('Train loss vs. Val loss')
    ax.show()
    return fig

def draw_acc(acc):
    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(x=list(range(args.epochs * 2))[::2], y=acc, color='b', title='Validation Accuracy')