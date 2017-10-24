"""RumourEval: Determining rumour veracity and support for rumours."""

import argparse
from classification.sdqc import sdqc
from classification.veracity_prediction import veracity_prediction
from scoring.Scorer import Scorer
from util.data import import_data
from util.log import setup_logger


DATASOURCES = ['dev', 'train', 'test']


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description='RumourEval, by Tong Liu and Joseph Roque')
    parser.add_argument('datasource', metavar='data', type=str, choices=DATASOURCES,
                        help='datasource for validation. \'dev\', \'train\', or \'test\'')
    parser.add_argument('--verbose', action='store_true',
                        help='enable verbose logging')
    return parser.parse_args()


def main(args):
    """Execute RumourEval program."""
    tweets = import_data(args.datasource)

    task_a_results = sdqc(tweets)
    task_b_results = veracity_prediction(tweets)

    task_a_scorer = Scorer('A', args.datasource)
    task_a_scorer.score(task_a_results)

    task_b_scorer = Scorer('B', args.datasource)
    task_b_scorer.score(task_b_results)


if __name__ == "__main__":
    ARGS = parse_args()
    setup_logger(ARGS.verbose)
    main(ARGS)
