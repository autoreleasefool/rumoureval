"""Package for classifying tweets by Support, Deny, Query, or Comment (SDQC)."""

import logging
import re
from time import time
from nltk.stem.porter import PorterStemmer
from nltk.tokenize.casual import TweetTokenizer, URLS
import numpy as np
from sklearn import metrics
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from ..pipeline.item_selector import ItemSelector
from ..util.log import get_log_separator


CLASSES = ['support', 'deny', 'query', 'comment']
LOGGER = logging.getLogger()
URLS_RE = re.compile(r"""(%s)""" % URLS, re.VERBOSE | re.I | re.UNICODE)


class TweetDetailExtractor(BaseEstimator, TransformerMixin):
    """Extract relevant details from tweets."""
    # pylint:disable=C0103,W0613,R0201

    def fit(self, x, y=None):
        """Fit to data."""
        return self

    def transform(self, tweets):
        """Transform a list of tweets to a set of attributes that sklearn can utilize.

        :param tweets:
            tweets to transform
        :type tweets:
            `list` of :class:`Tweet`
        :rtype:
            :class:`np.recarray`
        """
        features = np.recarray(shape=(len(tweets),),
                               dtype=[('text', object),
                                      ('verified', bool),
                                      ('hashtags', list),
                                      ('user_mentions', list),
                                      ('retweet_count', int),
                                      ('depth', int)])

        for i, tweet in enumerate(tweets):
            features['text'][i] = tweet['text']
            features['verified'][i] = tweet['user']['verified']
            features['hashtags'][i] = tweet['entities']['hashtags']
            features['user_mentions'][i] = tweet['entities']['user_mentions']
            features['retweet_count'][i] = tweet['retweet_count']
            features['depth'][i] = 0

            parent = tweet.parent()
            while parent != None:
                features['depth'][i] += 1
                parent = parent.parent()

        return features


class StemmingCountVectorizer(CountVectorizer):
    """CountVectorizer which counts occurrences of stemmed words in a document."""
    # pylint:disable=W0622,R0913,R0914

    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), analyzer='word',
                 max_df=1.0, min_df=1, max_features=None,
                 vocabulary=None, binary=False, dtype=np.int64):
        super(StemmingCountVectorizer, self).__init__(
            input=input, encoding=encoding,
            decode_error=decode_error, strip_accents=strip_accents,
            lowercase=lowercase, preprocessor=preprocessor, tokenizer=tokenizer,
            stop_words=stop_words, token_pattern=token_pattern,
            ngram_range=ngram_range, analyzer=analyzer,
            max_df=max_df, min_df=min_df, max_features=max_features,
            vocabulary=vocabulary, binary=binary, dtype=dtype)
        self._stemmer = PorterStemmer()
        self._tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)


    def _stem(self, tokens):
        """
        Stem a list of tokens to their roots.

        :param tokens:
            words to stem
        :type tokens:
            `list` of `str`
        :rtype:
            :class:`Generator` of `str`
        """
        for token in tokens:
            if token[0] == '#':
                # yield token
                pass
            elif token[0] == '@':
                # yield token
                pass
            else:
                yield self._stemmer.stem(token)


    def build_tokenizer(self):
        """Return a function that splits a string into a list of tokens."""
        return lambda doc: list(self._stem([
            token for token in self._tokenizer.tokenize(doc) if not URLS_RE.match(token)]))


class VerificationChecker(BaseEstimator, TransformerMixin):
    """Checks for various attributes of the tweet."""
    # pylint:disable=C0103,W0613,R0201

    def fit(self, x, y=None):
        """Fit to data."""
        return self

    def transform(self, tweets):
        """Transform a list of tweets to a set of attributes that sklearn can utilize.

        :param tweets:
            tweets to transform
        :type tweets:
            `list` of :class:`Tweet`
        :rtype:
            `dict`
        """
        return [{'verified': verified_status}
                for verified_status in tweets]


class FeatureCounter(BaseEstimator, TransformerMixin):
    """Count properties in the text of the tweet"""
    # pylint:disable=C0103,W0613,R0201

    def __init__(self, name):
        """Name the feature to count.

        :param name:
            name of the feature
        :type name:
            `str`
        """
        self.name = name

    def fit(self, x, y=None):
        """Fit to data."""
        return self

    def transform(self, tweets):
        """Transform a list of tweets to a set of attributes that sklearn can utilize.

        :param tweets:
            tweets to transform
        :type tweets:
            `list` of :class:`Tweet`
        :rtype:
            `dict`
        """
        return [{'{}_count'.format(self.name): len(feature) if isinstance(feature, list) else feature}
                for feature in tweets]


def sdqc(tweets_train, tweets_eval, train_annotations, eval_annotations):
    """
    Classify tweets into one of four categories - support (s), deny (d), query(q), comment (c).

    :param tweets_train:
        set of twitter threads to train model on
    :type tweets_train:
        `list` of :class:`Tweet`
    :param tweets_eval:
        set of twitter threads to evaluate model on
    :type tweets_eval:
        `list` of :class:`Tweet`
    :param train_annotations:
        sqdc task annotations for training data
    :type train_annotations:
        `list` of `str`
    :param eval_annotations:
        sqdc task annotations for evaluation data
    :type eval_annotations:
        `list` of `str`
    :rtype:
        `dict`
    """
    # pylint:disable=too-many-locals
    LOGGER.info(get_log_separator())
    LOGGER.info('Beginning SDQC Task (Task A)')

    LOGGER.info('Initializing pipeline')
    pipeline = Pipeline([
        # Extract useful features from tweets
        ('extract_tweets', TweetDetailExtractor()),

        # Combine processing of features
        ('union', FeatureUnion(
            transformer_list=[

                # Count occurrences on tweet text
                ('tweet_text', Pipeline([
                    ('selector', ItemSelector(key='text')),
                    ('count', StemmingCountVectorizer(stop_words='english')),
                ])),

                # Check if the tweeter is verified
                ('user_verified', Pipeline([
                    ('selector', ItemSelector(key='verified')),
                    ('verification', VerificationChecker()),
                    ('vect', DictVectorizer())
                ])),

                # Count occurrences of hashtags
                ('hashtags', Pipeline([
                    ('selector', ItemSelector(key='hashtags')),
                    ('count', FeatureCounter(name='hashtags')),
                    ('vect', DictVectorizer())
                ])),

                # Count occurrences of user mentions
                ('user_mentions', Pipeline([
                    ('selector', ItemSelector(key='user_mentions')),
                    ('count', FeatureCounter(name='user_mentions')),
                    ('vect', DictVectorizer())
                ])),

                # Count number of retweets
                ('retweet_count', Pipeline([
                    ('selector', ItemSelector(key='retweet_count')),
                    ('count', FeatureCounter(name='retweet_count')),
                    ('vect', DictVectorizer())
                ])),

                # Count depth of tweet
                ('tweet_depth', Pipeline([
                    ('selector', ItemSelector(key='depth')),
                    ('count', FeatureCounter(name='depth')),
                    ('vect', DictVectorizer())
                ])),

            ],

            # Relative weights of transformations
            transformer_weights={
                'tweet_text': 1.0,
                'user_verified': 1.0,
                'hashtags': 0.5,
                'user_mentions': 0.5,
                'retweet_count': 0.5,
                'tweet_depth': 0.5,
            },

        )),

        # Use a classifier on the result
        ('classifier', MultinomialNB())

        ])
    LOGGER.info(pipeline)

    y_train = [train_annotations[x['id_str']] for x in tweets_train]
    y_eval = [eval_annotations[x['id_str']] for x in tweets_eval]

    # Training on tweets_train
    start_time = time()
    pipeline.fit(tweets_train, y_train)
    LOGGER.info("")
    LOGGER.debug("train time: %0.3fs", time() - start_time)

    # Predicting classes for tweets_eval
    start_time = time()
    predictions = pipeline.predict(tweets_eval)
    LOGGER.debug("eval time:  %0.3fs", time() - start_time)

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
