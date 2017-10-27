"""RumourEval: Determining rumour veracity and support for rumours."""

import argparse
from src.classification.sdqc import sdqc
from src.classification.veracity_prediction import veracity_prediction
from src.scoring.Scorer import Scorer
from src.util.data import import_data, import_annotation_data
from src.util.log import setup_logger


LOGGER = None


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description='RumourEval, by Tong Liu and Joseph Roque')
    parser.add_argument('--test', action='store_true',
                        help='run with test data. defaults to run with dev data')
    parser.add_argument('--verbose', action='store_true',
                        help='enable verbose logging')
    return parser.parse_args()


def main(args):
    """Execute RumourEval program."""
    eval_datasource = 'test' if args.test else 'dev'

    # Import training and evaluation datasets
    tweets_train = import_data('train')
    tweets_eval = import_data(eval_datasource)

    # Import annotation data for training and evaluation datasets
    train_annotations = import_annotation_data('train')
    eval_annotations = import_annotation_data(eval_datasource)

    # Get the root tweets for each dataset for veracity prediction
    root_tweets_train = [x for x in tweets_train if x.is_source]
    root_tweets_eval = [x for x in tweets_eval if x.is_source]

    # Perform sdqc task
    task_a_results = sdqc(tweets_train,
                          tweets_eval,
                          train_annotations[0],
                          eval_annotations[0])

    # Perform veracity prediction task
    task_b_results = veracity_prediction(root_tweets_train,
                                         root_tweets_eval,
                                         train_annotations[1],
                                         eval_annotations[1])

    # Score tasks and output results
    task_a_scorer = Scorer('A', eval_datasource)
    task_a_scorer.score(task_a_results)

    task_b_scorer = Scorer('B', eval_datasource)
    task_b_scorer.score(task_b_results)

    LOGGER.info('')


if __name__ == "__main__":
    ARGS = parse_args()
    LOGGER = setup_logger(ARGS.verbose)
    main(ARGS)
