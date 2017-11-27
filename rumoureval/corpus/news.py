"""Information about news organizations."""

# Set of identifiers which might indicate a user/screen name belongs to a news organization
NEWS_IDENTIFIERS = frozenset([
    'new',
    'news',
    'ctv',
    'cnn',
    'cbc',
    'afp',
    'bbc',
    'wsj',
    'fox',
    'abc',
    'aj',
    'nbc',
])


def is_news(text):
    """Given a string, return True or False based on the likelihood that the text
    is closely associated with a news organization.

    :param text:
        a piece of text
    :type text:
        `str`
    """
    text = text.lower()
    for i in range(len(text) - 2):
        if text[i: i + 2] in NEWS_IDENTIFIERS or (
                (i + 2 < len(text) and text[i : i + 3] in NEWS_IDENTIFIERS) or (
                    (i + 3 < len(text) and text[i : i + 4] in NEWS_IDENTIFIERS))):
            return True
    return False
