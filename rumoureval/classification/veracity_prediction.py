"""Package for predicting veracity of tweets."""

import logging
from time import time
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import BernoulliNB
from .classify import benchmark
from ..util.data import size_mb
from ..util.log import get_log_separator


CLASSES = ['true', 'false', 'unverified']
LOGGER = logging.getLogger()


def veracity_prediction(tweets_train, tweets_eval, train_annotations, eval_annotations):
    """
    Predict the veracity of tweets.

    :param tweets_train:
        set of twitter threads to train model on
    :type tweets_train:
        `list` of :class:`Tweet`
    :param tweets_eval:
        set of twitter threads to evaluate model on
    :type tweets_eval:
        `list` of :class:`Tweet`
    :param train_annotations:
        veracity prediction task annotations for training data
    :type train_annotations:
        `list` of `str`
    :param eval_annotations:
        veracity prediction task annotations for evaluation data
    :type eval_annotations:
        `list` of `str`
    :rtype:
        `dict`
    """
    # pylint:disable=too-many-locals
    LOGGER.info(get_log_separator())
    LOGGER.info('Beginning Veracity Prediction Task (Task B)')

    # Convert training data to documents for bag of words
    training_docs = [x['text'] for x in tweets_train]
    training_doc_data_size_mb = size_mb(training_docs)
    y_train = [train_annotations[x['id_str']] for x in tweets_train]

    # Convert evaluation data to documents for bag of words
    eval_docs = [x['text'] for x in tweets_eval]
    eval_doc_data_size_mb = size_mb(eval_docs)
    y_eval = [eval_annotations[x['id_str']] for x in tweets_eval]

    # Time bag of words vectorization of training data
    LOGGER.debug("Extracting features from the training data using a sparse vectorizer")
    start_time = time()
    vectorizer = HashingVectorizer(stop_words='english', alternate_sign=False, n_features=2 ** 16)
    x_train = vectorizer.transform(training_docs)
    duration = time() - start_time
    LOGGER.debug("done in %fs at %0.3fMB/s", duration, training_doc_data_size_mb / duration)
    LOGGER.debug("n_samples: %d, n_features: %d", x_train.shape[0], x_train.shape[1])

    # Time bag of words vectorization of eval data
    LOGGER.debug("Extracting features from the eval data using the same vectorizer")
    start_time = time()
    x_eval = vectorizer.transform(eval_docs)
    duration = time() - start_time
    LOGGER.debug("done in %fs at %0.3fMB/s", duration, eval_doc_data_size_mb / duration)
    LOGGER.debug("n_samples: %d, n_features: %d", x_eval.shape[0], x_eval.shape[1])

    # Perform classification
    lst_results = benchmark(BernoulliNB(alpha=.01), x_train, y_train, x_eval, y_eval, CLASSES)

    # Convert results to dict of tweet ID to predicted class and confidence
    results = {}
    for (i, result) in enumerate(lst_results):
        results[tweets_eval[i]['id_str']] = (result, 1)

    return results
