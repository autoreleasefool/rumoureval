"""Extract relevant details from tweets."""

import dateutil.parser
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
    SWEAR_WORDS, RACES_RELIGIONS_POLITICAL
)
from ..corpus.stop_words import STOP_WORDS


URLS_RE = re.compile(r"""(%s)""" % URLS, re.VERBOSE | re.I | re.UNICODE)
PUNCTUATION_RE = re.compile(r'(\.)|(\?)|(\!)|(\.\.\.)|( )')
NON_ALPHA_RE = re.compile(r'^[^a-z]+$', re.I)

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
TWEET_DETAIL_CACHE = {
    'A': {},
    'B': {},
}

# Set of tweet details and the kind of detail
TWEET_DETAILS = [
    # Text properties
    ('text', str),
    ('text_stemmed', list),
    ('text_stemmed_stopped', list),
    ('text_minus_root', list),

    # Boolean properties
    ('verified', bool),
    ('is_news', bool),
    ('is_root', bool),
    ('has_url', bool),
    ('ends_with_question', bool),

    # Basic features
    ('hashtags', list),
    ('user_mentions', list),
    ('favorite_count', int),
    ('depth', int),
    ('retweet_count', int),
    ('account_age', int),

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
    ('number_count', int),

    # Child tweet properties
    ('child_denies', int),
    ('child_queries', int),
    ('child_comments', int),
    ('child_supports', int),

    # Percentage of sdq tweets
    ('support_percentage', float),
    ('denies_percentage', float),
    ('queries_percentage', float),
]

class TweetDetailExtractor(BaseEstimator, TransformerMixin):
    """Extract relevant details from tweets."""
    # pylint:disable=C0103,W0613,R0201

    def __init__(self, task='A', strip_hashtags=False, strip_mentions=False, classifications=None):
        """Initialize stemmer and tokenizer."""
        self._task = task
        self._tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
        self._strip_hashtags = strip_hashtags
        self._strip_mentions = strip_mentions
        self._classifications = classifications


    def get_params(self, deep=True):
        """Get the params"""
        return {
            'task': self._task,
            'strip_hashtags': self._strip_hashtags,
            'strip_mentions': self._strip_mentions,
            'classifications': self._classifications,
        }


    def set_params(self, **parameters):
        """Set the params"""
        for parameter, value in parameters.items():
            setattr(self, '_{}'.format(parameter), value)
        self._tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)


    @staticmethod
    def get_parseable_tweet_text(tweet, task='A'):
        """Given a tweet, return the most parseable tweet text.

        :param tweet:
            a tweet
        :type:
            :class:`Tweet`
        :param task:
            the task, 'A', or 'B'
        :type task:
            `str`
        :rtype:
            `str`
        """
        # Expanding tweet text for better accuracy
        if tweet['id'] in TWEET_DETAIL_CACHE[task]:
            return TWEET_DETAIL_CACHE[task][tweet['id']]['text']

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
        """
        pe = period
        qu = question mark
        ex = exclamation mart
        el = ellipsis
        sp = space
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
            if tweet['id'] in TWEET_DETAIL_CACHE[self._task]:
                properties = TWEET_DETAIL_CACHE[self._task][tweet['id']]
            else:
                expanded_text = TweetDetailExtractor.get_parseable_tweet_text(tweet, task=self._task)
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
                properties['has_url'] = 1 if URLS_RE.match(tweet['text']) else -1
                properties['favorite_count'] = tweet['favorite_count']

                account_created_at = dateutil.parser.parse(tweet['user']['created_at'])
                tweet_created_at = dateutil.parser.parse(tweet['created_at'])
                properties['account_age'] = (tweet_created_at - account_created_at).days

                # Get parent tweet
                depth = 0
                root = tweet
                while root.parent() is not None:
                    depth += 1
                    root = root.parent()
                properties['depth'] = depth

                # Boolean properties
                properties['is_news'] = 1 if is_news(tweet['user']['screen_name']) else -1
                properties['is_root'] = 1 if depth == 0 else -1
                properties['verified'] = 1 if tweet['user']['verified'] else -1

                # Get last term in the tweet
                last_term = None
                if len(properties['text_stemmed_stopped']) > 0:
                    last_term = -1
                    while abs(last_term - 1) < len(properties['text_stemmed_stopped']) and \
                        (properties['text_stemmed_stopped'][last_term][0] == '#' or \
                            NON_ALPHA_RE.match(properties['text_stemmed_stopped'][last_term][0])):
                        last_term -= 1
                    last_term = properties['text_stemmed_stopped'][last_term][-1]
                properties['ends_with_question'] = 1 if len(properties['text_stemmed_stopped']) > 0 and properties['text_stemmed_stopped'][-1][-1] == '?' else -1

                properties['text_minus_root'] = list(
                    set(properties['text_stemmed_stopped']) -
                    set(self._tokenize(TweetDetailExtractor.get_parseable_tweet_text(root, task=self._task)))
                )

                # Count the punctuations
                punc_count = self._count_punctuation(properties['text'])
                properties['period_count'] = punc_count['pe']
                properties['question_mark_count'] = punc_count['qu']
                properties['exclamation_count'] = punc_count['ex']
                properties['ellipsis_count'] = punc_count['el']

                # Count the characters in the tweet, minus spaces
                properties['char_count'] = len(properties['text']) - punc_count['sp']
                properties['number_count'] = len([
                    number for number in properties['text_stemmed_stopped'] if re.match(r'[0-9]+', number)
                ])

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

                if self._task == 'B':
                    properties['child_denies'] = len([
                        tweet for tweet in self._classifications if self._classifications[tweet] == 'deny'
                    ])
                    properties['child_queries'] = len([
                        tweet for tweet in self._classifications if self._classifications[tweet] == 'query'
                    ])
                    properties['child_comments'] = len([
                        tweet for tweet in self._classifications if self._classifications[tweet] == 'comment'
                    ])
                    properties['child_supports'] = len([
                        tweet for tweet in self._classifications if self._classifications[tweet] == 'support'
                    ])

                    total_sdq_tweets = properties['child_supports'] + properties['child_denies'] + properties['child_queries']
                    properties['support_percentage'] = properties['child_supports'] / total_sdq_tweets
                    properties['denies_percentage'] = properties['child_denies'] / total_sdq_tweets
                    properties['queries_percentage'] = properties['child_queries'] / total_sdq_tweets
                else:
                    properties['child_denies'] = 0
                    properties['child_queries'] = 0
                    properties['child_comments'] = 0
                    properties['child_supports'] = 0
                    properties['support_percentage'] = 0
                    properties['denies_percentage'] = 0
                    properties['queries_percentage'] = 0

            for detail in TWEET_DETAILS:
                features[detail[0]][i] = properties[detail[0]]

            # Cache the generated details for the tweet
            TWEET_DETAIL_CACHE[self._task][tweet['id']] = properties

        return features
