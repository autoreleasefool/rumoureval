"""Provide logging utilities."""

import logging
import sys


def get_log_separator(thick=True):
    """Returns a log separator string.

    :rtype:
        `str`
    """
    return '{}{}{}'.format(
        '\n' if thick else '',
        ('=' if thick else '-') * 30,
        '\n' if thick else ''
    )


def setup_logger(debug=False):
    """Sets up the logging state."""
    logger = logging.getLogger()
    if debug:
        logger_format = '%(levelname)s (%(module)s#%(lineno)d) %(message)s'
        logging.basicConfig(format=logger_format)
    else:
        sys.stderr = None
        logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    return logger
