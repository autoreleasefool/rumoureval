"""Extract relevant details from tweets."""

import re
from html import unescape
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.tokenize.casual import TweetTokenizer, URLS
from sklearn.base import BaseEstimator, TransformerMixin
from ..corpus.opinion import (
    POSITIVE_WORDS, NEGATIVE_WORDS, QUERYING_WORDS, DENYING_WORDS,
    POSITIVE_ACRONYMS, NEGATIVE_ACRONYMS, QUERYING_ACRONYMS, DENYING_ACRONYMS,
    POSITIVE_EMOJI, NEGATIVE_EMOJI, QUERYING_EMOJI, DENYING_EMOJI,
    SWEAR_WORDS, RACES_RELIGIONS_POLITICAL
)
from ..corpus.stop_words import STOP_WORDS
from ..corpus.contractions import CONTRACTIONS


URLS_RE = re.compile(r"""(%s)""" % URLS, re.VERBOSE | re.I | re.UNICODE)

STEMMER = PorterStemmer()
STEMMED_STOP_WORDS = frozenset([STEMMER.stem(word) for word in STOP_WORDS])
STEMMED_LEXICON = {
    'positive': frozenset([STEMMER.stem(word) for word in POSITIVE_WORDS]),
    'negative': frozenset([STEMMER.stem(word) for word in NEGATIVE_WORDS]),
    'querying': frozenset([STEMMER.stem(word) for word in QUERYING_WORDS]),
    'denying': frozenset([STEMMER.stem(word) for word in DENYING_WORDS]),
    'swear': frozenset([STEMMER.stem(word) for word in SWEAR_WORDS]),
    'personal': frozenset([STEMMER.stem(word) for word in RACES_RELIGIONS_POLITICAL]),
}


class TweetDetailExtractor(BaseEstimator, TransformerMixin):
    """Extract relevant details from tweets."""
    # pylint:disable=C0103,W0613,R0201

    def __init__(self, strip_hashtags=True, strip_mentions=True):
        """Initialize stemmer and tokenizer."""
        self._tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
        self._strip_hashtags = strip_hashtags
        self._strip_mentions = strip_mentions

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
            if self._strip_hashtags and token[0] == '#':
                pass
            elif self._strip_mentions and token[0] == '@':
                pass
            else:
                yield STEMMER.stem(token)

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
                                      ('text_stemmed_stopped', list),
                                      ('verified', bool),
                                      ('hashtags', list),
                                      ('user_mentions', list),
                                      ('retweet_count', int),
                                      ('depth', int),
                                      ('positive_words', list),
                                      ('negative_words', list),
                                      ('querying_words', list),
                                      ('denying_words', list),
                                      ('swear_words', list),
                                      ('personal_words', list)])

        for i, tweet in enumerate(tweets):
            # Expanding tweet text for better accuracy
            expanded_text = unescape(tweet['text'])
            expanded_text = expanded_text.split(' ')
            expanded_text = [
                CONTRACTIONS[word] if word in CONTRACTIONS else word for word in expanded_text
                ]
            expanded_text = ' '.join(expanded_text)
            features['text'][i] = expanded_text

            # Stem, and remove stop words
            stemmed = self._tokenize(expanded_text)
            features['text_stemmed'][i] = stemmed
            features['text_stemmed_stopped'][i] = [
                w for w in stemmed if w not in STEMMED_STOP_WORDS
                ]

            # Basic features
            features['verified'][i] = 1 if tweet['user']['verified'] else 0
            features['hashtags'][i] = tweet['entities']['hashtags']
            features['user_mentions'][i] = tweet['entities']['user_mentions']
            features['retweet_count'][i] = tweet['retweet_count']
            features['depth'][i] = 0

            # Sentiment analysis
            features['positive_words'][i] = [
                word for word in stemmed if word in STEMMED_LEXICON['positive']
                ]
            features['negative_words'][i] = [
                word for word in stemmed if word in STEMMED_LEXICON['negative']
                ]
            features['querying_words'][i] = [
                word for word in stemmed if word in STEMMED_LEXICON['querying']
                ]
            features['denying_words'][i] = [
                word for word in stemmed if word in STEMMED_LEXICON['denying']
                ]
            features['swear_words'][i] = [
                word for word in stemmed if word in STEMMED_LEXICON['swear']
                ]
            features['personal_words'][i] = [
                word for word in stemmed if word in STEMMED_LEXICON['personal']
                ]

            parent = tweet.parent()
            while parent is not None:
                features['depth'][i] += 1
                parent = parent.parent()

        return features
