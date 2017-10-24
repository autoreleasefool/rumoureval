"""Provide logging utilities."""

import logging
import sys


def get_log_separator():
    """Returns a log separator string.

    :rtype:
        `str`
    """
    return '-' * 30


def setup_logger(debug=False):
    """Sets up the logging state."""
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    return logger
