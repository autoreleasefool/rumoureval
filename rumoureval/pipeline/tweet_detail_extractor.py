"""Extract relevant details from tweets."""

import re
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.tokenize.casual import TweetTokenizer, URLS
from sklearn.base import BaseEstimator, TransformerMixin


URLS_RE = re.compile(r"""(%s)""" % URLS, re.VERBOSE | re.I | re.UNICODE)


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
