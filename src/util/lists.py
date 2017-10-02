"""
List manipulation utilities
"""

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
