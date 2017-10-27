"""Provide logging utilities."""

import logging


def get_log_separator():
    """Returns a log separator string.

    :rtype:
        `str`
    """
    return '-' * 30


def setup_logger(debug=False):
    """Sets up the logging state."""
    if debug:
        logger_format = '%(levelname)s (%(module)s#%(lineno)d) %(message)s'
        logging.basicConfig(format=logger_format)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    return logger
