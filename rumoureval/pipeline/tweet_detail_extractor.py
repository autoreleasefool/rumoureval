"""Extract relevant details from tweets."""

import re
from html import unescape
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.tokenize.casual import TweetTokenizer, URLS
from sklearn.base import BaseEstimator, TransformerMixin
from ..corpus.contractions import CONTRACTIONS
from ..corpus.news import is_news
from ..corpus.opinion import (
    POSITIVE_WORDS, NEGATIVE_WORDS, QUERYING_WORDS, DENYING_WORDS,
    POSITIVE_ACRONYMS, NEGATIVE_ACRONYMS, QUERYING_ACRONYMS, DENYING_ACRONYMS,
    POSITIVE_EMOJI, NEGATIVE_EMOJI, QUERYING_EMOJI, DENYING_EMOJI,
    SWEAR_WORDS, RACES_RELIGIONS_POLITICAL
)
from ..corpus.stop_words import STOP_WORDS


URLS_RE = re.compile(r"""(%s)""" % URLS, re.VERBOSE | re.I | re.UNICODE)
PUNCTUATION_RE = re.compile(r'(\.)|(\?)|(\!)|(\.\.\.)|( )')

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

# Cache tweet details constructed to save computation time for multiple pipelines
TWEET_DETAIL_CACHE = {}

# Set of tweet details and the kind of detail
TWEET_DETAILS = [
    # Text properties
    ('text', str),
    ('text_stemmed', list),
    ('text_stemmed_stopped', list),
    ('text_minus_root', list),

    # Boolean properties
    ('verified', int),
    ('is_news', int),
    ('is_root', int),

    # Basic features
    ('hashtags', list),
    ('user_mentions', list),
    ('depth', int),
    ('retweet_count', int),

    # Sentimental analysis
    ('positive_words', list),
    ('negative_words', list),
    ('querying_words', list),
    ('denying_words', list),
    ('swear_words', list),
    ('personal_words', list),

    # Punctuation
    ('period_count', int),
    ('question_mark_count', int),
    ('exclamation_count', int),
    ('ellipsis_count', int),
    ('char_count', int),
]


class TweetDetailExtractor(BaseEstimator, TransformerMixin):
    """Extract relevant details from tweets."""
    # pylint:disable=C0103,W0613,R0201

    def __init__(self, strip_hashtags=True, strip_mentions=True):
        """Initialize stemmer and tokenizer."""
        self._tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
        self._strip_hashtags = strip_hashtags
        self._strip_mentions = strip_mentions


    @staticmethod
    def get_parseable_tweet_text(tweet):
        """Given a tweet, return the most parseable tweet text.

        :param tweet:
            a tweet
        :type:
            :class:`Tweet`
        :rtype:
            `str`
        """
        # Expanding tweet text for better accuracy
        if tweet['id'] in TWEET_DETAIL_CACHE:
            return TWEET_DETAIL_CACHE[tweet['id']]['text']

        expanded_text = tweet['text'].encode('ascii', 'ignore').decode('ascii')
        expanded_text = unescape(expanded_text)
        expanded_text = expanded_text.split(' ')
        expanded_text = [
            CONTRACTIONS[word] if word in CONTRACTIONS else word for word in expanded_text
            ]
        expanded_text = ' '.join(expanded_text)
        return expanded_text


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

    def _count_punctuation(self, tweet):
        """
        Count the number of punctuations. Unfortunately, since I'm using regex, the ordering matters because
        of the grouping order

        :param tweet:
            tweet body
        :type tweet:
            `str`
        :rtype:
            `dict`
        """
        res = {
            'pe': 0,
            'qu': 0,
            'ex': 0,
            'el': 0,
            'sp': 0,
        }

        for match_group in PUNCTUATION_RE.findall(tweet):
            if match_group[0]:
                res['pe'] += 1
            if match_group[1]:
                res['qu'] += 1
            if match_group[2]:
                res['ex'] += 1
            if match_group[3]:
                res['el'] += 1
            if match_group[4]:
                res['sp'] += 1

        return res

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
                               dtype=TWEET_DETAILS)

        for i, tweet in enumerate(tweets):
            # Check if the details have been calculated before, and pull from cache if so
            properties = {}
            if tweet['id'] in TWEET_DETAIL_CACHE:
                properties = TWEET_DETAIL_CACHE[tweet['id']]
            else:
                expanded_text = TweetDetailExtractor.get_parseable_tweet_text(tweet)
                properties['text'] = expanded_text

                # Stem, and remove stop words
                stemmed = self._tokenize(expanded_text)
                properties['text_stemmed'] = stemmed
                properties['text_stemmed_stopped'] = [
                    w for w in stemmed if w not in STEMMED_STOP_WORDS
                    ]

                # Basic features
                properties['hashtags'] = tweet['entities']['hashtags']
                properties['user_mentions'] = tweet['entities']['user_mentions']
                properties['retweet_count'] = tweet['retweet_count']

                # Get parent tweet
                depth = 0
                root = tweet
                while root.parent() is not None:
                    depth += 1
                    root = root.parent()
                properties['depth'] = depth

                # Boolean properties
                properties['is_news'] = 1 if is_news(tweet['user']['screen_name']) else 0
                properties['is_root'] = 1 if depth == 0 else 0
                properties['verified'] = 1 if tweet['user']['verified'] else 0

                properties['text_minus_root'] = list(
                    set(properties['text_stemmed_stopped']) -
                    set(self._tokenize(TweetDetailExtractor.get_parseable_tweet_text(root)))
                )

                # Count the punctuations
                punc_count = self._count_punctuation(properties['text'])
                properties['period_count'] = punc_count['pe']
                properties['question_mark_count'] = punc_count['qu']
                properties['exclamation_count'] = punc_count['ex']
                properties['ellipsis_count'] = punc_count['el']

                # Count the characters in the tweet, minus spaces
                properties['char_count'] = len(properties['text']) - punc_count['sp']

                properties['is_news'] = 1 if is_news(tweet['user']['screen_name']) else 0
                properties['is_root'] = 0 if depth == 0 else 1

                # Sentiment analysis
                properties['positive_words'] = [
                    word for word in stemmed if word in STEMMED_LEXICON['positive']
                    ]
                properties['negative_words'] = [
                    word for word in stemmed if word in STEMMED_LEXICON['negative']
                    ]
                properties['querying_words'] = [
                    word for word in stemmed if word in STEMMED_LEXICON['querying']
                    ]
                properties['denying_words'] = [
                    word for word in stemmed if word in STEMMED_LEXICON['denying']
                    ]
                properties['swear_words'] = [
                    word for word in stemmed if word in STEMMED_LEXICON['swear']
                    ]
                properties['personal_words'] = [
                    word for word in stemmed if word in STEMMED_LEXICON['personal']
                    ]

            for detail in TWEET_DETAILS:
                features[detail[0]][i] = properties[detail[0]]

            # Cache the generated details for the tweet
            TWEET_DETAIL_CACHE[tweet['id']] = properties

        return features
