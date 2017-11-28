"""
List manipulation utilities
"""

import itertools


def filter_none(base):
    """
    Filter items from a list which are None

    :param base:
        A list of items
    :type base:
        `list`
    :rtype:
        `list`
    """
    return [x for x in base if x is not None]


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


def dict_product(dicts):
    """
    >>> list(dict_product(dict(number=[1,2], character='ab')))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))