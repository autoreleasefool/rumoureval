"""Count number of words with varying sentiments."""

from sklearn.base import BaseEstimator, TransformerMixin


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
