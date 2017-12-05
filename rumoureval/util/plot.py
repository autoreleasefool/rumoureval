"""Plot results."""

import itertools
import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    From http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fg_color = 'white'
    # bg_color = 'black'


    im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    im.axes.tick_params(color=fg_color, labelcolor=fg_color)
    for spine in im.axes.spines.values():
        spine.set_edgecolor(fg_color)

    label = plt.title(title)
    label.set_color('white')

    cb = plt.colorbar()
    cb.set_label('colorbar label', color=fg_color)

    # set colorbar tick color
    cb.ax.yaxis.set_tick_params(color=fg_color)

    # set colorbar edgecolor
    cb.outline.set_edgecolor(fg_color)

    # set colorbar ticklabels
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=fg_color)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    label = plt.ylabel('True label')
    label.set_color('white')
    [i.set_color('white') for i in plt.gca().get_yticklabels()]

    label = plt.xlabel('Predicted label')
    label.set_color('white')
    [i.set_color('white') for i in plt.gca().get_xticklabels()]

    plt.savefig('{}.png'.format(title), transparent=True)