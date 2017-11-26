"""RumourEval: Determining rumour veracity and support for rumours."""

import argparse
import sys
import nltk
from .classification.sdqc import sdqc
from .classification.veracity_prediction import veracity_prediction
from .scoring.Scorer import Scorer
from .util.data import import_data, import_annotation_data, output_data_by_class
from .util.log import setup_logger


def main(args=None):
    """The main routine."""
    if args is None:
        args = sys.argv[1:]

    ######################
    # Set up Environment #
    ######################
    parser = argparse.ArgumentParser(description='RumourEval, by Tong Liu and Joseph Roque')
    parser.add_argument('--test', action='store_true',
                        help='run with test data. defaults to run with dev data')
    parser.add_argument('--verbose', action='store_true',
                        help='enable verbose logging')
    parser.add_argument('--osorted', action='store_true',
                        help='output tweets sorted by class')
    parsed_args = parser.parse_args()
    eval_datasource = 'test' if parsed_args.test else 'dev'

    # Setup logger
    logger = setup_logger(parsed_args.verbose)

    # Download NLTK corpus
    nltk.download('opinion_lexicon', quiet=True)

    ########################
    # Begin classification #
    ########################

    # Import training and evaluation datasets
    tweets_train = import_data('train')
    tweets_eval = import_data(eval_datasource)

    # Import annotation data for training and evaluation datasets
    train_annotations = import_annotation_data('train')
    eval_annotations = import_annotation_data(eval_datasource)

    # Output tweets sorted by class
    if parsed_args.osorted:
        output_data_by_class(tweets_train, train_annotations[0])

    # Get the root tweets for each dataset for veracity prediction
    root_tweets_train = [x for x in tweets_train if x.is_source]
    root_tweets_eval = [x for x in tweets_eval if x.is_source]

    # Output tweets sorted by class
    if parsed_args.osorted:
        output_data_by_class(root_tweets_train, train_annotations[1])

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

    logger.info('')


if __name__ == "__main__":
    main()
