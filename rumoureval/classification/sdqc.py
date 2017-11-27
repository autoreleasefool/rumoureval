"""Package for classifying tweets by Support, Deny, Query, or Comment (SDQC)."""

import logging
from time import time
from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from ..pipeline.item_selector import ItemSelector
from ..pipeline.feature_counter import FeatureCounter
from ..pipeline.pipelinize import pipelinize
from ..pipeline.tweet_detail_extractor import TweetDetailExtractor
from ..util.lists import list_to_str
from ..util.log import get_log_separator


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


def sdqc(tweets_train, tweets_eval, train_annotations, eval_annotations):
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
    :rtype:
        `dict`
    """
    # pylint:disable=too-many-locals
    LOGGER.info(get_log_separator())
    LOGGER.info('Beginning SDQC Task (Task A)')

    LOGGER.info('Filter tweets from training set')
    tweets_train = filter_tweets(tweets_train)

    LOGGER.info('Initializing pipeline')
    pipeline = Pipeline([
        # Extract useful features from tweets
        ('extract_tweets', TweetDetailExtractor(strip_hashtags=True, strip_mentions=True)),

        # Combine processing of features
        ('union', FeatureUnion(
            transformer_list=[

                # Count occurrences on tweet text
                ('tweet_text', Pipeline([
                    ('selector', ItemSelector(keys='text_stemmed_stopped')),
                    ('list_to_str', pipelinize(list_to_str)),
                    ('count', CountVectorizer()),
                ])),

                # Count numeric properties of the tweets
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

                ('count_depth', Pipeline([
                    ('selector', ItemSelector(keys='depth')),
                    ('count', FeatureCounter(names='depth')),
                    ('vect', DictVectorizer()),
                ])),

                ('verified', Pipeline([
                    ('selector', ItemSelector(keys='verified')),
                    ('count', FeatureCounter(names='verified')),
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
                'count_hashtags': 0.5,
                'count_mentions': 0.5,
                'count_retweets': 0.5,
                'count_depth': 0.5,
                'verified': 0.5,
                'pos_neg_sentiment': 1.0,
                'denying_words': 10.0,
                'querying_words': 10.0,
                'offensiveness': 10.0,
            },

        )),

        # Use a classifier on the result
        ('classifier', MultinomialNB())

        ])
    LOGGER.info(pipeline)

    y_train = [train_annotations[x['id_str']] for x in tweets_train]
    y_eval = [eval_annotations[x['id_str']] for x in tweets_eval]

    LOGGER.info('Beginning training')

    # Training on tweets_train
    start_time = time()
    pipeline.fit(tweets_train, y_train)
    LOGGER.info("")
    LOGGER.debug("train time: %0.3fs", time() - start_time)

    LOGGER.info('Beginning evaluation')

    # Predicting classes for tweets_eval
    start_time = time()
    predictions = pipeline.predict(tweets_eval)
    LOGGER.debug("eval time:  %0.3fs", time() - start_time)

    LOGGER.info('Completed SDQC Task (Task A). Printing results')

    # Outputting classifier results
    LOGGER.info("accuracy:   %0.3f", metrics.accuracy_score(y_eval, predictions))
    LOGGER.info("classification report:")
    LOGGER.info(metrics.classification_report(y_eval, predictions, target_names=CLASSES))
    LOGGER.info("confusion matrix:")
    LOGGER.info(metrics.confusion_matrix(y_eval, predictions))

    # Uncomment to see vocabulary
    # LOGGER.info(pipeline.get_params()['union__tweet_text__count'].get_feature_names())

    # Convert results to dict of tweet ID to predicted class
    results = {}
    for (i, prediction) in enumerate(predictions):
        results[tweets_eval[i]['id_str']] = prediction

    return results
