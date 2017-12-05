"""Package for classifying tweets by Support, Deny, Query, or Comment (SDQC)."""

import logging
import os
from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC
from sklearn.pipeline import FeatureUnion, Pipeline
from ..pipeline.item_selector import ItemSelector
from ..pipeline.feature_counter import FeatureCounter
from ..pipeline.pipelinize import pipelinize
from ..pipeline.tweet_detail_extractor import TweetDetailExtractor
from ..util.lists import list_to_str
from ..util.log import get_log_separator
from ..util.data import get_output_path
from ..util.plot import plot_confusion_matrix


LOGGER = logging.getLogger()
CLASSES = ['comment', 'deny', 'query', 'support']


def filter_tweets(tweets, filter_short=False, similarity_threshold=0.9):
    """Filter tweets which are believed to cause additional confusion in the classifier.

    :param tweets:
        list of twitter threads to train model on
    :type tweets:
        `list` of :class:`Tweet`
    :param filter_short:
        True to filter tweets which are too short to be meaningful
    :type filter_short:
        `bool`
    :param similarity_threshold:
        filter tweets which are this similar or more to their root tweet
    :type similarity_threshold:
        `float`
    :rtype:
        `list` of :class:`Tweet`
    """
    # Cached root tweet text
    root_cache = {}

    filtered_tweets = []
    for tweet in tweets:
        root_tweet = tweet
        while root_tweet.parent() is not None:
            root_tweet = root_tweet.parent()

        # Root tweets should not be filtered
        if root_tweet == tweet:
            filtered_tweets.append(tweet)
            continue

        # Get text of tweet and root tweet
        root_text = root_cache[root_tweet['id']] if root_tweet['id'] in root_cache else (
            TweetDetailExtractor.get_parseable_tweet_text(root_tweet)
        )
        root_cache[root_tweet['id']] = root_text

        tweet_text = TweetDetailExtractor.get_parseable_tweet_text(tweet)

        # Discard training tweet if too short
        if filter_short and len(tweet_text.split(' ')) < 3:
            continue

        # Calculate cosine similarity between tweet and root and discard if too similar
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform((root_text, tweet_text))

        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
        if similarity[0][1] < similarity_threshold:
            filtered_tweets.append(tweet)

    return filtered_tweets


def sdqc(tweets_train, tweets_eval, train_annotations, eval_annotations, use_cache, plot):
    """
    Classify tweets into one of four categories - support (s), deny (d), query(q), comment (c).

    :param tweets_train:
        list of twitter threads to train model on
    :type tweets_train:
        `list` of :class:`Tweet`
    :param tweets_eval:
        set of twitter threads to evaluate model on
    :type tweets_eval:
        `list` of :class:`Tweet`
    :param train_annotations:
        sqdc task annotations for training data
    :type train_annotations:
        `dict
    :param eval_annotations:
        sqdc task annotations for evaluation data
    :type eval_annotations:
        `dict`
    :param use_cache:
        true to enable using cached classifier
    :type use_cache:
        `bool`
    :param plot:
        true to plot confusion matrix
    :type plot:
        `bool`
    :rtype:
        `dict`
    """
    # pylint:disable=too-many-locals
    LOGGER.info(get_log_separator())
    LOGGER.info('Beginning SDQC Task (Task A)')

    LOGGER.info('Filter tweets from training set')
    tweets_train = filter_tweets(tweets_train)

    LOGGER.info('Initializing pipeline')

    LOGGER.info('Query pipeline')
    query_pipeline = build_query_pipeline()
    query_annotations = generate_one_vs_rest_annotations(train_annotations, 'query')
    eval_annotations_query = generate_one_vs_rest_annotations(eval_annotations, 'query')
    LOGGER.info(query_pipeline)

    LOGGER.info('Base pipeline')
    base_pipeline = build_base_pipeline()
    LOGGER.info(base_pipeline)

    y_train_base = [train_annotations[x['id_str']] for x in tweets_train]
    y_train_query = [query_annotations[x['id_str']] for x in tweets_train]
    y_eval_base = [eval_annotations[x['id_str']] for x in tweets_eval]
    y_eval_query = [eval_annotations_query[x['id_str']] for x in tweets_eval]

    LOGGER.info('Beginning training')

    # Training on tweets_train
    start_time = time()
    if use_cache and os.path.exists(os.path.join(get_output_path(), 'base_pipeline.pickle')):
        base_pipeline = joblib.load(os.path.join(get_output_path(), 'base_pipeline.pickle'))
    else:
        base_pipeline.fit(tweets_train, y_train_base)
        joblib.dump(base_pipeline, os.path.join(get_output_path(), 'base_pipeline.pickle'))
    LOGGER.info("base_pipeline training:  %0.3fs", time() - start_time)

    start_time = time()
    if use_cache and os.path.exists(os.path.join(get_output_path(), 'query_pipeline.pickle')):
        query_pipeline = joblib.load(os.path.join(get_output_path(), 'query_pipeline.pickle'))
    else:
        query_pipeline.fit(tweets_train, y_train_query)
        joblib.dump(query_pipeline, os.path.join(get_output_path(), 'query_pipeline.pickle'))
    LOGGER.info("query_pipeline training: %0.3fs", time() - start_time)

    LOGGER.info("")
    LOGGER.info('Beginning evaluation')

    # Predicting classes for tweets_eval
    start_time = time()
    base_predictions = base_pipeline.predict(tweets_eval)
    query_predictions = query_pipeline.predict(tweets_eval)

    # Boosting
    predictions = []
    for i in range(len(base_predictions)):
        if query_predictions[i] == 'query':
            predictions.append('query')
        else:
            predictions.append(base_predictions[i])

    LOGGER.debug("eval time:  %0.3fs", time() - start_time)

    LOGGER.debug('============================================================')
    LOGGER.debug('|                                                          |')
    LOGGER.debug('|                  Misclassified - query                   |')
    LOGGER.debug('|                                                          |')
    LOGGER.debug('============================================================')

    # Print misclassified query vs not_query
    for i, prediction in enumerate(query_predictions):
        if (prediction == 'query' and y_eval_base[i] != 'query') or (prediction == 'not_query' and y_eval_base[i] == 'query'):
            root = tweets_eval[i]
            while root.parent() != None:
                root = root.parent()
            LOGGER.debug('{}\t{}\t{}\n\t\t{}'.format(
                y_eval_base[i],
                prediction,
                TweetDetailExtractor.get_parseable_tweet_text(tweets_eval[i]),
                TweetDetailExtractor.get_parseable_tweet_text(root)
                ))

    LOGGER.info('Completed SDQC Task (Task A). Printing results')

    # Outputting classifier results
    LOGGER.info("query_accuracy:   %0.3f", metrics.accuracy_score(y_eval_query, query_predictions))
    LOGGER.info("base accuracy:    %0.3f", metrics.accuracy_score(y_eval_base, base_predictions))
    LOGGER.info("accuracy:         %0.3f", metrics.accuracy_score(y_eval_base, predictions))
    LOGGER.info("classification report (query):")
    LOGGER.info(metrics.classification_report(y_eval_query, query_predictions, target_names=['not_query', 'query']))
    LOGGER.info("classification report (base):")
    LOGGER.info(metrics.classification_report(y_eval_base, base_predictions, target_names=CLASSES))
    LOGGER.info("classification report (combined):")
    LOGGER.info(metrics.classification_report(y_eval_base, predictions, target_names=CLASSES))
    LOGGER.info("confusion matrix (query):")
    LOGGER.info(metrics.confusion_matrix(y_eval_query, query_predictions))
    LOGGER.info("confusion matrix (base):")
    LOGGER.info(metrics.confusion_matrix(y_eval_base, base_predictions))
    LOGGER.info("confusion matrix (combined):")
    LOGGER.info(metrics.confusion_matrix(y_eval_base, predictions))

    if plot:
        cm = metrics.confusion_matrix(y_eval_base, predictions)
        np.set_printoptions(precision=2)

        # Plot normalized confusion matrix
        figure = plt.figure()
        figure.set_facecolor('#1c212b')
        plot_confusion_matrix(cm, classes=CLASSES, normalize=True, title='Task A Confusion Matrix')

        plt.show()

    # Convert results to dict of tweet ID to predicted class
    results = {}
    for (i, prediction) in enumerate(predictions):
        results[tweets_eval[i]['id_str']] = prediction

    return results


def generate_one_vs_rest_annotations(annotations, one):
    """Convert annotation labels into a set of class vs not class.

    :param annotations:
        set of annotations for tweet IDs
    :type annotations
        `dict`
    :param one:
        the one annotation vs rest
    :type one:
        `str`
    """
    one_vs_rest_annotations = {}
    for tweet_id in annotations:
        one_vs_rest_annotations[tweet_id] = \
                annotations[tweet_id] if annotations[tweet_id] == one else 'not_{}'.format(one)
    return one_vs_rest_annotations


def build_query_pipeline():
    """Build a pipeline for predicting if a tweet is classified as query or not."""
    return Pipeline([
        # Extract useful features from tweets
        ('extract_tweets', TweetDetailExtractor(task='A', strip_hashtags=False, strip_mentions=False)),

        # Combine processing of features
        ('union', FeatureUnion(
            transformer_list=[

                # Count features
                ('count_depth', Pipeline([
                    ('selector', ItemSelector(keys='depth')),
                    ('count', FeatureCounter(names='depth')),
                    ('vect', DictVectorizer()),
                ])),

                # Boolean features
                ('is_news', Pipeline([
                    ('selector', ItemSelector(keys='is_news')),
                    ('count', FeatureCounter(names='is_news')),
                    ('vect', DictVectorizer()),
                ])),

                ('is_root', Pipeline([
                    ('selector', ItemSelector(keys='is_root')),
                    ('count', FeatureCounter(names='is_root')),
                    ('vect', DictVectorizer()),
                ])),

                ('ends_with_question', Pipeline([
                    ('selector', ItemSelector(keys='ends_with_question')),
                    ('count', FeatureCounter(names='ends_with_question')),
                    ('vect', DictVectorizer()),
                ])),

                # Punctuation
                ('count_question_marks', Pipeline([
                    ('selector', ItemSelector(keys='question_mark_count')),
                    ('count', FeatureCounter(names='question_mark_count')),
                    ('vect', DictVectorizer()),
                ])),

                # Count positive and negative words in the tweets
                ('pos_neg_sentiment', Pipeline([
                    ('selector', ItemSelector(keys=['positive_words', 'negative_words'])),
                    ('count', FeatureCounter(names=['positive_words', 'negative_words'])),
                    ('vect', DictVectorizer()),
                ])),

                # Count querying words in the tweets
                ('querying_words', Pipeline([
                    ('selector', ItemSelector(keys='querying_words')),
                    ('count', FeatureCounter(names='querying_words')),
                    ('vect', DictVectorizer()),
                ])),

            ],

            # Relative weights of transformations
            transformer_weights={
                'count_depth': 1.0,

                'is_news': 1.0,
                'is_root': 2.5,

                'count_question_marks': 5.0,

                'pos_neg_sentiment': 0.5,
                'querying_words': 1.0,
                'ends_with_question': 10.0,
            }

        )),

        # Use a classifier on the result
        ('classifier', SVC(C=1, kernel='linear', class_weight='balanced'))

    ])


def build_base_pipeline():
    """Build a pipeline for predicting all 4 SDQC classes."""
    return Pipeline([
        # Extract useful features from tweets
        ('extract_tweets', TweetDetailExtractor(task='A', strip_hashtags=False, strip_mentions=False)),

        # Combine processing of features
        ('union', FeatureUnion(
            transformer_list=[

                # Count occurrences on tweet text
                ('tweet_text', Pipeline([
                    ('selector', ItemSelector(keys='text_stemmed_stopped')),
                    ('list_to_str', pipelinize(list_to_str)),
                    ('count', TfidfVectorizer()),
                ])),

                # Boolean features
                ('is_news', Pipeline([
                    ('selector', ItemSelector(keys='is_news')),
                    ('count', FeatureCounter(names='is_news')),
                    ('vect', DictVectorizer()),
                ])),

                ('is_root', Pipeline([
                    ('selector', ItemSelector(keys='is_root')),
                    ('count', FeatureCounter(names='is_root')),
                    ('vect', DictVectorizer()),
                ])),

                ('verified', Pipeline([
                    ('selector', ItemSelector(keys='verified')),
                    ('count', FeatureCounter(names='verified')),
                    ('vect', DictVectorizer()),
                ])),

                ('ends_with_question', Pipeline([
                    ('selector', ItemSelector(keys='ends_with_question')),
                    ('count', FeatureCounter(names='ends_with_question')),
                    ('vect', DictVectorizer()),
                ])),

                # Punctuation
                ('count_periods', Pipeline([
                    ('selector', ItemSelector(keys='period_count')),
                    ('count', FeatureCounter(names='period_count')),
                    ('vect', DictVectorizer()),
                ])),

                ('count_question_marks', Pipeline([
                    ('selector', ItemSelector(keys='question_mark_count')),
                    ('count', FeatureCounter(names='question_mark_count')),
                    ('vect', DictVectorizer()),
                ])),

                ('count_exclamations', Pipeline([
                    ('selector', ItemSelector(keys='exclamation_count')),
                    ('count', FeatureCounter(names='exclamation_count')),
                    ('vect', DictVectorizer()),
                ])),

                ('count_ellipsis', Pipeline([
                    ('selector', ItemSelector(keys='ellipsis_count')),
                    ('count', FeatureCounter(names='ellipsis_count')),
                    ('vect', DictVectorizer()),
                ])),

                ('count_chars', Pipeline([
                    ('selector', ItemSelector(keys='char_count')),
                    ('count', FeatureCounter(names='char_count')),
                    ('vect', DictVectorizer()),
                ])),

                # Count features
                ('count_depth', Pipeline([
                    ('selector', ItemSelector(keys='depth')),
                    ('count', FeatureCounter(names='depth')),
                    ('vect', DictVectorizer()),
                ])),

                ('count_hashtags', Pipeline([
                    ('selector', ItemSelector(keys='hashtags')),
                    ('count', FeatureCounter(names='hashtags')),
                    ('vect', DictVectorizer()),
                ])),

                ('count_mentions', Pipeline([
                    ('selector', ItemSelector(keys='user_mentions')),
                    ('count', FeatureCounter(names='user_mentions')),
                    ('vect', DictVectorizer()),
                ])),

                ('count_retweets', Pipeline([
                    ('selector', ItemSelector(keys='retweet_count')),
                    ('count', FeatureCounter(names='retweet_count')),
                    ('vect', DictVectorizer()),
                ])),

                # Count positive and negative words in the tweets
                ('pos_neg_sentiment', Pipeline([
                    ('selector', ItemSelector(keys=['positive_words', 'negative_words'])),
                    ('count', FeatureCounter(names=['positive_words', 'negative_words'])),
                    ('vect', DictVectorizer()),
                ])),

                # Count denying words in the tweets
                ('denying_words', Pipeline([
                    ('selector', ItemSelector(keys='denying_words')),
                    ('count', FeatureCounter(names='denying_words')),
                    ('vect', DictVectorizer()),
                ])),

                # Count querying words in the tweets
                ('querying_words', Pipeline([
                    ('selector', ItemSelector(keys='querying_words')),
                    ('count', FeatureCounter(names='querying_words')),
                    ('vect', DictVectorizer()),
                ])),

                # Count swear words and personal attacks
                ('offensiveness', Pipeline([
                    ('selector', ItemSelector(keys=['swear_words', 'personal_words'])),
                    ('count', FeatureCounter(names=['swear_words', 'personal_words'])),
                    ('vect', DictVectorizer()),
                ])),

            ],

            # Relative weights of transformations
            transformer_weights={
                'tweet_text': 1.0,

                'verified': 0.5,
                'is_news': 5.0,
                'is_root': 20.0,
                'ends_with_question': 10.0,

                'count_periods': 0.5,
                'count_question_marks': 0.5,
                'count_exclamations': 0.5,
                'count_chars': 0.5,

                'count_hashtags': 0.5,
                'count_mentions': 0.5,
                'count_retweets': 0.5,
                'count_depth': 0.5,

                'pos_neg_sentiment': 1.0,
                'denying_words': 1.0,
                'querying_words': 1.0,
                'offensiveness': 5.0,
            },

        )),

        # Use a classifier on the result
        ('classifier', SVC(C=100, gamma=0.001, kernel='rbf'))

    ])