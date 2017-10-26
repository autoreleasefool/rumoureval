"""Package for predicting veracity of tweets."""

import logging
from time import time

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import BernoulliNB

from classification.classify import benchmark
from util.data import size_mb


LOGGER = logging.getLogger()


def veracity_prediction(tweets_train, tweets_test, train_annotations, test_annotations):
    """
    Predict the veracity of tweets.

    :param tweets_train:
        set of twitter threads to train on
    :type tweets_train:
        `list` of :class:`Tweet`
    :param tweets_test:
        set of twitter threads to test model on
    :type tweets_test:
        `list` of :class:`Tweet`
    :param train_annotations:
        veracity_predication task annotations for training data
    :type train_annotations:
        `list` of `str`
    :param test_annotations:
        veracity_predication task annotations for testing data
    :type test_annotations:
        `list` of `str`
    :rtype:
        `dict`
    """
    tweet_training_data = [x['text'] for x in tweets_train]
    tweet_training_data_size_mb = size_mb(tweet_training_data)
    y_train = [train_annotations[x['id_str']] for x in tweets_train]

    tweet_testing_data = [x['text'] for x in tweets_test]
    tweet_testing_data_size_mb = size_mb(tweet_testing_data)
    y_test = [test_annotations[x['id_str']] for x in tweets_test]

    LOGGER.info("Extracting features from the training data using a sparse vectorizer")
    start_time = time()
    vectorizer = HashingVectorizer(stop_words='english', alternate_sign=False,
                                   n_features=2 ** 16)
    x_train = vectorizer.transform(tweet_training_data)
    duration = time() - start_time
    LOGGER.info("done in %fs at %0.3fMB/s" % (duration, tweet_training_data_size_mb / duration))
    LOGGER.info("n_samples: %d, n_features: %d" % x_train.shape)
    LOGGER.info("")

    LOGGER.info("Extracting features from the test data using the same vectorizer")
    start_time = time()
    x_test = vectorizer.transform(tweet_testing_data)
    duration = time() - start_time
    LOGGER.info("done in %fs at %0.3fMB/s" % (duration, tweet_testing_data_size_mb / duration))
    LOGGER.info("n_samples: %d, n_features: %d" % x_test.shape)
    LOGGER.info("")

    print('=' * 80)
    print("Naive Bayes")
    lst_results = benchmark(BernoulliNB(alpha=.01), x_train, y_train, x_test, y_test)

    results = {}
    for (i, result) in enumerate(lst_results):
        results[tweets_test[i]['id_str']] = (result, 1)

    return results
