"""RumourEval: Determining rumour veracity and support for rumours."""

import argparse
import logging
from util.data import import_data
from util.log import setup_logger

LOGGER = None
DATASOURCES = ['dev', 'train', 'test']

def import_tweets(datasource):
    """
    Import tweet data from the given datasource
    :param datasource:
        Data source to use to build and test ML algorithm
    :type datasource:
        'dev', 'train', or 'test'
    """
    tweets, (task_a_annotations, task_b_annotations) = import_data(datasource)

    if LOGGER.getEffectiveLevel() == logging.DEBUG:
        LOGGER.debug('Imported %d root tweets from %s', len(tweets), datasource)
        LOGGER.debug('Imported %d annotations for subtask B', len(task_b_annotations.keys()))

        # Count number of tweets and children
        tweets_to_iterate = tweets[:]
        total_tweets = 0
        for tweet in tweets_to_iterate:
            total_tweets += 1
            tweets_to_iterate += list(tweet.children())

        LOGGER.debug('Imported %d child tweets from %s', total_tweets, datasource)
        LOGGER.debug('Imported %d annotations for subtask A', len(task_a_annotations.keys()))


def main():
    """Parse arguments and execute program."""
    global LOGGER # pylint:disable=W0603
    parser = argparse.ArgumentParser(description='RumourEval, by Tong Liu and Joseph Roque')
    parser.add_argument('datasource', metavar='data', type=str, choices=DATASOURCES,
                        help='datasource for validation. \'dev\', \'train\', or \'test\'')
    parser.add_argument('--verbose', action='store_true',
                        help='enable verbose logging')

    args = parser.parse_args()

    LOGGER = setup_logger(args.verbose)
    import_tweets(args.datasource)


if __name__ == "__main__":
    main()
