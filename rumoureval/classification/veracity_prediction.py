"""Package for predicting veracity of tweets."""

import logging
from time import time
from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import FeatureUnion, Pipeline
from ..pipeline.item_selector import ItemSelector
from ..pipeline.feature_counter import FeatureCounter
from ..pipeline.tweet_detail_extractor import TweetDetailExtractor
from ..pipeline.pipelinize import pipelinize
from ..util.lists import list_to_str
from ..util.log import get_log_separator


CLASSES = ['false', 'true', 'unverified']
LOGGER = logging.getLogger()


def veracity_prediction(tweets_train, tweets_eval, train_annotations, eval_annotations, task_a_results):
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
    :param task_a_results:
        classification results from task A
    :type task_a_results:
        `dict`
    :rtype:
        `dict`
    """
    # pylint:disable=too-many-locals
    LOGGER.info(get_log_separator())
    LOGGER.info('Beginning Veracity Prediction Task (Task B)')

    LOGGER.info('Initializing pipeline')
    pipeline = Pipeline([
        # Extract useful features from tweets
        ('extract_tweets', TweetDetailExtractor(task='B', strip_hashtags=False, strip_mentions=False, classifications=task_a_results)),

        # Combine processing of features
        ('union', FeatureUnion(
            transformer_list=[

                 # Count occurrences on tweet text
                ('tweet_text', Pipeline([
                    ('selector', ItemSelector(keys='text_stemmed_stopped')),
                    ('list_to_str', pipelinize(list_to_str)),
                    ('count', TfidfVectorizer()),
                ])),

                # Percentages of support, deny and query tweets
                ('percentage_of_support', Pipeline([
                    ('selector', ItemSelector(keys='support_percentage')),
                    ('count', FeatureCounter(names='support_percentage')),
                    ('vect', DictVectorizer()),
                ])),
                ('percentage_of_denies', Pipeline([
                    ('selector', ItemSelector(keys='denies_percentage')),
                    ('count', FeatureCounter(names='denies_percentage')),
                    ('vect', DictVectorizer()),
                ])),
                ('percentage_of_queries', Pipeline([
                    ('selector', ItemSelector(keys='queries_percentage')),
                    ('count', FeatureCounter(names='queries_percentage')),
                    ('vect', DictVectorizer()),
                ])),

            ],

            # Relative weights of transformations
            transformer_weights={
                'tweet_text': 1.0,
                'percentage_of_support': 1.0,
                'percentage_of_denies': 1.0,
                'percentage_of_queries': 1.0,
            },
        )),

        # Use a classifier on the result
        # ('classifier', BernoulliNB(alpha=0.1))
        ('classifier', SVC(kernel='rbf'))

        ])
    LOGGER.info(pipeline)

    y_train = [train_annotations[x['id_str']] for x in tweets_train]
    y_eval = [eval_annotations[x['id_str']] for x in tweets_eval]

    # Training on tweets_train
    start_time = time()
    pipeline.fit(tweets_train, y_train)
    LOGGER.debug("train time: %0.3fs", time() - start_time)

    # Predicting classes for tweets_eval
    start_time = time()
    predictions = pipeline.predict(tweets_eval)
    LOGGER.debug("eval time:  %0.3fs", time() - start_time)

    # Outputting classifier results
    LOGGER.debug("accuracy:   %0.3f", metrics.accuracy_score(y_eval, predictions))
    LOGGER.info("\nclassification report:")
    LOGGER.info(metrics.classification_report(y_eval, predictions, target_names=CLASSES))
    LOGGER.info("confusion matrix:")
    LOGGER.info(metrics.confusion_matrix(y_eval, predictions))

    # Convert results to dict of tweet ID to predicted class
    results = {}
    for (i, prediction) in enumerate(predictions):
        results[tweets_eval[i]['id_str']] = (prediction, 1)

    return results
