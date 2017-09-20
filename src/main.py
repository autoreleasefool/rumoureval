"""RumourEval: Determining rumour veracity and support for rumours."""

import argparse
from util.data import import_data
from util.log import setup_logger

DATASOURCES = ['dev', 'train', 'test']

def import_tweets(datasource):
    """
    Import tweet data from the given datasource
    :param datasource:
        Data source to use to build and test ML algorithm
    :type datasource:
        'dev', 'train', or 'test'
    """
    import_data(datasource)

def main():
    """Parse arguments and execute program."""
    parser = argparse.ArgumentParser(description='RumourEval, by Tong Liu and Joseph Roque')
    parser.add_argument('datasource', metavar='data', type=str, choices=DATASOURCES,
                        help='datasource for validation. \'dev\', \'train\', or \'test\'')
    parser.add_argument('--verbose', action='store_true',
                        help='enable verbose logging')

    args = parser.parse_args()

    setup_logger(args.verbose)
    import_tweets(args.datasource)

if __name__ == "__main__":
    main()
