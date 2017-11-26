"""Tweet properties and context"""


class Tweet(object):
    """
    Contains details about a single Tweet, including the poster, the context,
    and references to any parents or children.
    """


    def __init__(self, raw_tweet, children=None, parent=None, is_source=False):
        self._raw_tweet = raw_tweet
        self._children = children
        self._parent = parent
        self.is_source = is_source

        for child in self._children:
            child._parent = self  # pylint:disable=W0212


    def children(self):
        """
        Get the child tweets of this tweet.

        :rtype:
            Generator[:class:`Tweet`]
        """
        for child in self._children:
            yield child


    def parent(self):
        """
        Get the parent tweet of this tweet.

        :rtype:
            :class:`Tweet` or None
        """
        return self._parent


    def raw(self):
        """Get the raw tweet JSON.

        :rtype:
            json
        """
        return self._raw_tweet


    def __getitem__(self, name):
        return self._raw_tweet[name] if name in self._raw_tweet else None


    def __str__(self):
        return str(self._raw_tweet)
