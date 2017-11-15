"""Package for classifying tweets by Support, Deny, Query, or Comment (SDQC)."""

import logging
import re
from time import time
from nltk.corpus import opinion_lexicon
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
from ..pipeline.pipelinize import pipelinize
from ..util.log import get_log_separator


LOGGER = logging.getLogger()
CLASSES = ['support', 'deny', 'query', 'comment']

URLS_RE = re.compile(r"""(%s)""" % URLS, re.VERBOSE | re.I | re.UNICODE)


def get_sentimental_lexicons(stemmed=False):
    """
    Get words for sentiment analysis.

    :param stemmed:
        True to stem the lexicons before returning
    :type stemmed:
        `bool`
    :rtype:
        `dict`
    """
    sentiment_attributes = {
        'positive': set(opinion_lexicon.positive()),
        'negative': set(opinion_lexicon.negative()),
    }

    if not stemmed:
        return sentiment_attributes

    stemmer = PorterStemmer()
    for sentiment in sentiment_attributes:
        stemmed = set()
        for word in sentiment_attributes[sentiment]:
            stemmed.add(stemmer.stem(word))
        sentiment_attributes[sentiment] = stemmed
    return sentiment_attributes


def list_to_str(lst):
    """Convert a list of values to a space-delimited string.

    :param lst:
        the list to convert
    :type lst:
        `list`
    :rtype:
        `str`
    """
    return ' '.join(lst)


class TweetDetailExtractor(BaseEstimator, TransformerMixin):
    """Extract relevant details from tweets."""
    # pylint:disable=C0103,W0613,R0201

    def __init__(self):
        """Initialize stemmer and tokenizer."""
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

    def _tokenize(self, tweet):
        """Split a tweet body into a list of tokens.
        :param tweet:
            tweet body
        :type tweet:
            `str`
        :rtype:
            `list` of `str`
        """
        return list(self._stem([
            token for token in self._tokenizer.tokenize(tweet) if not URLS_RE.match(token)]))

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
                               dtype=[('text', str),
                                      ('text_stemmed', list),
                                      ('verified', bool),
                                      ('hashtags', list),
                                      ('user_mentions', list),
                                      ('retweet_count', int),
                                      ('depth', int)])

        for i, tweet in enumerate(tweets):
            features['text'][i] = tweet['text']
            features['text_stemmed'][i] = self._tokenize(tweet['text'])
            features['verified'][i] = 1 if tweet['user']['verified'] else 0
            features['hashtags'][i] = tweet['entities']['hashtags']
            features['user_mentions'][i] = tweet['entities']['user_mentions']
            features['retweet_count'][i] = tweet['retweet_count']
            features['depth'][i] = 0

            parent = tweet.parent()
            while parent != None:
                features['depth'][i] += 1
                parent = parent.parent()

        return features


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
            `list`
        """
        return [{'verified': verified_status}
                for verified_status in tweets]


class FeatureCounter(BaseEstimator, TransformerMixin):
    """Count properties in the text of the tweet"""
    # pylint:disable=C0103,W0613,R0201

    def __init__(self, names):
        """Names the features to count.

        :param names:
            names of the features
        :type names:
            `list` of `str`
        """
        self.names = names

    def fit(self, x, y=None):
        """Fit to data."""
        return self

    def transform(self, tweets_features):
        """Transform a list of features to a set of attributes that sklearn can utilize.

        :param tweets_features:
            tweet features to transform
        :type tweets_features:
            `list` of `dict`
        :rtype:
            `list`
        """
        transformed = []
        for name in self.names:
            for i in range(len(tweets_features[name])):
                if len(transformed) <= i:
                    transformed.append({})
                transformed[i][name] = len(tweets_features[name][i]) if \
                        isinstance(tweets_features[name][i], list) else tweets_features[name][i]
        return transformed


class SentimentalCounter(BaseEstimator, TransformerMixin):
    """Count number of words with varying sentiments."""
    # pylint:disable=C0103,W0613,R0201

    def __init__(self, lexicons):
        """Stores the sentimental lexicons to later apply to the tweets.

        :param lexicons:
            set of sentiment lexicons
        :type lexicons:
            `dict`
        """
        self.lexicons = lexicons

    def fit(self, x, y=None):
        """Fit to data."""
        return self

    def transform(self, tweets):
        """Transform a tweet to a set of sentiment counts that sklearn can utilize.

        :param tweets:
            tweets to transform
        :type tweets:
            `list` of `text`
        :rtype:
            `list`
        """
        transformed = []
        for tweet in tweets:
            sentiment_counts = {}
            for sentiment in self.lexicons:
                count = len([word for word in tweet if word in self.lexicons[sentiment]])
                sentiment_counts[sentiment] = count
            transformed.append(sentiment_counts)
        return transformed


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

    LOGGER.info('Retrieve stemmed corpus for later usage')
    sentimental_lexicons = get_sentimental_lexicons(stemmed=True)

    LOGGER.info('Initializing pipeline')
    pipeline = Pipeline([
        # Extract useful features from tweets
        ('extract_tweets', TweetDetailExtractor()),

        # Combine processing of features
        ('union', FeatureUnion(
            transformer_list=[

                # Count occurrences on tweet text
                ('tweet_text', Pipeline([
                    ('selector', ItemSelector(keys='text_stemmed')),
                    ('list_to_str', pipelinize(list_to_str)),
                    ('count', CountVectorizer(stop_words='english')),
                ])),

                # Count numeric properties of the tweets
                ('count', Pipeline([

                    ('selector', ItemSelector(keys=[
                        'hashtags',
                        'user_mentions',
                        'retweet_count',
                        'depth',
                        'verified',
                    ])),

                    ('count', FeatureCounter(names=[
                        'hashtags',
                        'user_mentions',
                        'retweet_count',
                        'depth',
                        'verified',
                    ])),

                    ('vect', DictVectorizer()),
                ])),

                # Count positive and negative words in the tweets
                ('sentiment', Pipeline([
                    ('selector', ItemSelector(keys='text_stemmed')),
                    ('count', SentimentalCounter(lexicons=sentimental_lexicons)),
                    ('vect', DictVectorizer()),
                ]))

            ],

            # Relative weights of transformations
            transformer_weights={
                'tweet_text': 1.0,
                'count': 1.0,
                'sentiment': 0.5,
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
