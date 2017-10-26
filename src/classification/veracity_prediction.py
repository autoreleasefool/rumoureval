"""Package for predicting veracity of tweets."""

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
        sqdc task annotations for training data
    :type train_annotations:
        `list` of `str`
    :param test_annotations:
        sqdc task annotations for testing data
    :type test_annotations:
        `list` of `str`
    :rtype:
        `dict`
    """
    return {}
