"""Count properties in the text of tweets."""

from sklearn.base import BaseEstimator, TransformerMixin


class FeatureCounter(BaseEstimator, TransformerMixin):
    """Count properties in the text of the tweet"""
    # pylint:disable=C0103,W0613,R0201

    def __init__(self, names):
        """Names the features to count.

        :param names:
            names of the features
        :type names:
            `list` of `str` or `str`
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
        if isinstance(self.names, list):
            for name in self.names:
                for i in range(len(tweets_features[name])):
                    if len(transformed) <= i:
                        transformed.append({})
                    transformed[i][name] = len(tweets_features[name][i]) if \
                        isinstance(tweets_features[name][i], list) else tweets_features[name][i]
        else:
            name = self.names
            for feature in tweets_features:
                count = len(feature) if isinstance(feature, list) else feature
                transformed.append({'{}_count'.format(name): count})

        return transformed
