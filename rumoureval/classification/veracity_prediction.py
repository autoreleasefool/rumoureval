"""Package for predicting veracity of tweets."""

import logging
from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from ..pipeline.item_selector import ItemSelector
from ..pipeline.feature_counter import FeatureCounter
from ..pipeline.tweet_detail_extractor import TweetDetailExtractor
from ..pipeline.pipelinize import pipelinize
from ..util.lists import list_to_str
from ..util.log import get_log_separator
from ..util.plot import plot_confusion_matrix


CLASSES = ['false', 'true', 'unverified']
LOGGER = logging.getLogger()


def filter_tweets(tweets, annotations):
    """Filter tweets which are believed to cause additional confusion in the classifier.

    :param tweets:
        list of twitter threads to train model on
    :type tweets:
        `list` of :class:`Tweet`
    :param annotations:
        Mapping of tweet id to their annotations
    :type annotations:
        `dict`
    :rtype:
        `list` of :class:`Tweet`
    """
    filtered_tweets = []
    for tweet in tweets:
        if annotations[tweet['id_str']] == 'unverified':
            continue
        filtered_tweets.append(tweet)

    return filtered_tweets


def veracity_prediction(tweets_train, tweets_eval, train_annotations, eval_annotations, task_a_results, plot):
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
    :param plot:
        true to plot confusion matrix
    :type plot:
        `bool`
    :rtype:
        `dict`
    """
    # pylint:disable=too-many-locals
    LOGGER.info(get_log_separator())
    LOGGER.info('Beginning Veracity Prediction Task (Task B)')

    LOGGER.info('Filter tweets from training set')
    tweets_train = filter_tweets(tweets_train, train_annotations)

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

                # Count features
                ('number_count', Pipeline([
                    ('selector', ItemSelector(keys='number_count')),
                    ('count', FeatureCounter(names='number_count')),
                    ('vect', DictVectorizer()),
                ])),

                ('count_chars', Pipeline([
                    ('selector', ItemSelector(keys='char_count')),
                    ('count', FeatureCounter(names='char_count')),
                    ('vect', DictVectorizer()),
                ])),

                # Boolean features
                ('verified', Pipeline([
                    ('selector', ItemSelector(keys='verified')),
                    ('count', FeatureCounter(names='verified')),
                    ('vect', DictVectorizer()),
                ])),

                ('is_root', Pipeline([
                    ('selector', ItemSelector(keys='is_root')),
                    ('count', FeatureCounter(names='is_root')),
                    ('vect', DictVectorizer()),
                ])),

                ('has_url', Pipeline([
                    ('selector', ItemSelector(keys='has_url')),
                    ('count', FeatureCounter(names='has_url')),
                    ('vect', DictVectorizer()),
                ])),

            ],

            # Relative weights of transformations
            transformer_weights={

                # Bag of words
                'tweet_text': 2.0,

                # Percetange of child tweets
                'percentage_of_support': 1.0,
                'percentage_of_denies': 1.0,
                'percentage_of_queries': 1.0,

                # Count features
                'number_count': 1.0,
                'count_chars': 1.0,

                # Boolean features
                'verified': 1.0,
                'is_root': 1.5,
                'has_url': 1.0,

            },
        )),

        ('scaler', StandardScaler(with_mean=False)),

        # Use a classifier on the result
        ('classifier', SVC(kernel='rbf', class_weight='balanced', probability=True))

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
    confidence = pipeline.predict_proba(tweets_eval)
    LOGGER.debug("eval time:  %0.3fs", time() - start_time)

    # Print misclassified tweets
    LOGGER.debug('============================================================')
    LOGGER.debug('|                                                          |')
    LOGGER.debug('|                      Misclassified                       |')
    LOGGER.debug('|                                                          |')
    LOGGER.debug('============================================================')
    for i, prediction in enumerate(predictions):
        if prediction != y_eval[i]:
            LOGGER.debug('--------------------\n{}\n{}\n{}\n{}'.format(
                y_eval[i],
                prediction,
                confidence[i],
                TweetDetailExtractor.get_parseable_tweet_text(tweets_eval[i])
                ))

    # Outputting classifier results
    LOGGER.debug("accuracy:   %0.3f", metrics.accuracy_score(y_eval, predictions))
    LOGGER.info("\nclassification report:")
    LOGGER.info(metrics.classification_report(y_eval, predictions, target_names=CLASSES))
    LOGGER.info("confusion matrix:")
    LOGGER.info(metrics.confusion_matrix(y_eval, predictions))

    if plot:
        cm = metrics.confusion_matrix(y_eval, predictions)
        np.set_printoptions(precision=2)

        # Plot normalized confusion matrix
        figure = plt.figure()
        figure.set_facecolor('#1c212b')
        plot_confusion_matrix(cm, classes=CLASSES, normalize=True, title='Task B Confusion Matrix')

        plt.show()

    # Convert results to dict of tweet ID to predicted class
    results = {}
    for i, prediction in enumerate(predictions):
        prediction_confidence = max(confidence[i]) if max(confidence[i]) < 0.9 else 1
        results[tweets_eval[i]['id_str']] = (prediction, prediction_confidence)

    return results
