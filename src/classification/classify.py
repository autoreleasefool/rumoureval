"""Train and use a classifier."""

import logging
from time import time

from sklearn import metrics


LOGGER = logging.getLogger()


def benchmark(clf, x_train, y_train, x_eval, y_eval):
    """
    Train a classifier, then predict the classes for evaluation data.

    :param clf:
        classifier
    :type clf:
        :class:`Classifier`
    :param x_train:
        training dataset
    :type x_train:
        `list` of :class:`Tweet`
    :param y_train:
        training dataset labels
    :type y_train:
        `list` of `str`
    :param x_eval:
        evaluation dataset
    :type x_eval:
        `list` of :class:`Tweet`
    :param y_eval:
        evaluation dataset labels
    :type y_eval:
        `list` of `str`
    :rtype:
        `list` of `str`
    """
    LOGGER.debug('_' * 80)
    LOGGER.debug("Training: ")
    LOGGER.debug(clf)
    start_time = time()
    clf.fit(x_train, y_train)
    train_time = time() - start_time
    LOGGER.debug("train time: %0.3fs" % train_time)

    start_time = time()
    predictions = clf.predict(x_eval)
    eval_time = time() - start_time
    LOGGER.debug("eval time:  %0.3fs" % eval_time)

    score = metrics.accuracy_score(y_eval, predictions)
    LOGGER.debug("accuracy:   %0.3f" % score)
    return predictions
