"""Provide logging utilities."""

import logging

def setup_logger(debug=False):
    """Sets up the logging state."""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    return logger
