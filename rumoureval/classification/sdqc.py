"""Package for classifying tweets by Support, Deny, Query, or Comment (SDQC)."""

import logging
from time import time
from nltk.stem.porter import PorterStemmer
import numpy as np
from sklearn import metrics
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from ..pipeline.item_selector import ItemSelector
from ..util.log import get_log_separator


CLASSES = ['support', 'deny', 'query', 'comment']
LOGGER = logging.getLogger()


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
                               dtype=[('text', object)])

        for i, tweet in enumerate(tweets):
            features['text'][i] = tweet['text']

        return features


class StemmingCountVectorizer(CountVectorizer):
    """CountVectorizer which counts occurrences of stemmed words in a document."""

    def __init__(self, *args, **kwargs):
        try:
            super(StemmingCountVectorizer, self).__init__(*args, **kwargs)
        except RuntimeError:
            pass
        self._stemmer = PorterStemmer()


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
            yield self._stemmer.stem(token)


    def build_tokenizer(self):
        """Return a function that splits a string into a list of tokens."""
        tokenize = super(StemmingCountVectorizer, self).build_tokenizer()
        return lambda doc: list(self._stem(tokenize(doc)))


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

            ],

            # Relative weights of transformations
            transformer_weights={
                'tweet_text': 1.0,
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
        results[tweets_eval[i]['id_str']] = prediction

    return results
