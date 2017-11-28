"""Package for classifying tweets by Support, Deny, Query, or Comment (SDQC)."""

import logging
from time import time
from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC
from sklearn.pipeline import FeatureUnion, Pipeline
from ..pipeline.item_selector import ItemSelector
from ..pipeline.feature_counter import FeatureCounter
from ..pipeline.pipelinize import pipelinize
from ..pipeline.tweet_detail_extractor import TweetDetailExtractor
from ..util.lists import list_to_str
from ..util.log import get_log_separator


LOGGER = logging.getLogger()
CLASSES = ['comment', 'deny', 'query', 'support']


def filter_tweets(tweets, filter_short=False, similarity_threshold=0.9):
    """Filter tweets which are believed to cause additional confusion in the classifier.

    :param tweets:
        list of twitter threads to train model on
    :type tweets:
        `list` of :class:`Tweet`
    :param filter_short:
        True to filter tweets which are too short to be meaningful
    :type filter_short:
        `bool`
    :param similarity_threshold:
        filter tweets which are this similar or more to their root tweet
    :type similarity_threshold:
        `float`
    :rtype:
        `list` of :class:`Tweet`
    """
    # Cached root tweet text
    root_cache = {}

    filtered_tweets = []
    for tweet in tweets:
        root_tweet = tweet
        while root_tweet.parent() is not None:
            root_tweet = root_tweet.parent()

        # Root tweets should not be filtered
        if root_tweet == tweet:
            filtered_tweets.append(tweet)
            continue

        # Get text of tweet and root tweet
        root_text = root_cache[root_tweet['id']] if root_tweet['id'] in root_cache else (
            TweetDetailExtractor.get_parseable_tweet_text(root_tweet)
        )
        root_cache[root_tweet['id']] = root_text

        tweet_text = TweetDetailExtractor.get_parseable_tweet_text(tweet)

        # Discard training tweet if too short
        if filter_short and len(tweet_text.split(' ')) < 3:
            continue

        # Calculate cosine similarity between tweet and root and discard if too similar
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform((root_text, tweet_text))

        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
        if similarity[0][1] < similarity_threshold:
            filtered_tweets.append(tweet)

    return filtered_tweets


def sdqc(tweets_train, tweets_eval, train_annotations, eval_annotations):
    """
    Classify tweets into one of four categories - support (s), deny (d), query(q), comment (c).

    :param tweets_train:
        list of twitter threads to train model on
    :type tweets_train:
        `list` of :class:`Tweet`
    :param tweets_eval:
        set of twitter threads to evaluate model on
    :type tweets_eval:
        `list` of :class:`Tweet`
    :param train_annotations:
        sqdc task annotations for training data
    :type train_annotations:
        `dict
    :param eval_annotations:
        sqdc task annotations for evaluation data
    :type eval_annotations:
        `dict`
    :rtype:
        `dict`
    """
    # pylint:disable=too-many-locals
    LOGGER.info(get_log_separator())
    LOGGER.info('Beginning SDQC Task (Task A)')

    LOGGER.info('Filter tweets from training set')
    tweets_train = filter_tweets(tweets_train)

    LOGGER.info('Initializing pipeline')

    LOGGER.debug('Deny pipeline')
    deny_pipeline = build_deny_pipeline()
    deny_annotations = generate_one_vs_rest_annotations(train_annotations, 'deny')
    eval_annotations_deny = generate_one_vs_rest_annotations(eval_annotations, 'deny')
    LOGGER.info(deny_pipeline)

    LOGGER.debug('Query pipeline')
    query_pipeline = build_query_pipeline()
    query_annotations = generate_one_vs_rest_annotations(train_annotations, 'query')
    eval_annotations_query = generate_one_vs_rest_annotations(eval_annotations, 'query')
    LOGGER.info(query_pipeline)

    LOGGER.debug('Base pipeline')
    base_pipeline = build_base_pipeline()
    LOGGER.info(base_pipeline)

    y_train_base = [train_annotations[x['id_str']] for x in tweets_train]
    y_train_deny = [deny_annotations[x['id_str']] for x in tweets_train]
    y_train_query = [query_annotations[x['id_str']] for x in tweets_train]
    y_eval_base = [eval_annotations[x['id_str']] for x in tweets_eval]
    y_eval_deny = [eval_annotations_deny[x['id_str']] for x in tweets_eval]
    y_eval_query = [eval_annotations_query[x['id_str']] for x in tweets_eval]

    LOGGER.info('Beginning training')

    # Training on tweets_train
    start_time = time()

    LOGGER.info('Training base')
    base_pipeline.fit(tweets_train, y_train_base)

    LOGGER.info('Training deny')
    deny_pipeline.fit(tweets_train, y_train_deny)

    LOGGER.info('Training query')
    query_pipeline.fit(tweets_train, y_train_query)

    LOGGER.info("")
    LOGGER.debug("train time: %0.3fs", time() - start_time)

    LOGGER.info('Beginning evaluation')

    # Predicting classes for tweets_eval
    start_time = time()
    base_predictions = base_pipeline.predict(tweets_eval)
    deny_predictions = deny_pipeline.predict(tweets_eval)
    query_predictions = query_pipeline.predict(tweets_eval)

    print('============================================================')
    print('|                                                          |')
    print('|                  Misclassified - query                   |')
    print('|                                                          |')
    print('============================================================')

    # Print misclassified query vs not_query
    for i, prediction in enumerate(query_predictions):
        if (prediction == 'query' and y_eval_base[i] != 'query') or (prediction == 'not_query' and y_eval_base[i] == 'query'):
            root = tweets_eval[i]
            while root.parent() != None:
                root = root.parent()
            print('{}\t{}\t{}\n\t\t{}'.format(
                y_eval_base[i],
                prediction,
                TweetDetailExtractor.get_parseable_tweet_text(tweets_eval[i]),
                TweetDetailExtractor.get_parseable_tweet_text(root)
                ))

    print('============================================================')
    print('|                                                          |')
    print('|                  Misclassified - deny                    |')
    print('|                                                          |')
    print('============================================================')

    # Print misclassified deny vs not_deny
    for i, prediction in enumerate(deny_predictions):
        if (prediction == 'deny' and y_eval_base[i] != 'deny') or (prediction == 'not_deny' and y_eval_base[i] == 'deny'):
            root = tweets_eval[i]
            while root.parent() != None:
                root = root.parent()
            print('{}\t{}\t{}\n\t\t{}'.format(
                y_eval_base[i],
                prediction,
                TweetDetailExtractor.get_parseable_tweet_text(tweets_eval[i]),
                TweetDetailExtractor.get_parseable_tweet_text(root)
                ))

    predictions = []
    for i in range(len(base_predictions)):
        if query_predictions[i] == 'query':
            predictions.append('query')
        # elif deny_predictions[i] == 'deny':
        #     predictions.append('deny')
        else:
            predictions.append(base_predictions[i])

    LOGGER.debug("eval time:  %0.3fs", time() - start_time)
    LOGGER.info('Completed SDQC Task (Task A). Printing results')

    # Outputting classifier results
    LOGGER.info("deny_accuracy:    %0.3f", metrics.accuracy_score(y_eval_deny, deny_predictions))
    LOGGER.info("query_accuracy:   %0.3f", metrics.accuracy_score(y_eval_query, query_predictions))
    LOGGER.info("base accuracy:    %0.3f", metrics.accuracy_score(y_eval_base, base_predictions))
    LOGGER.info("accuracy:         %0.3f", metrics.accuracy_score(y_eval_base, predictions))
    LOGGER.info("classification report:")
    LOGGER.info(metrics.classification_report(y_eval_deny, deny_predictions, target_names=['deny', 'not_deny']))
    LOGGER.info(metrics.classification_report(y_eval_query, query_predictions, target_names=['not_query', 'query']))
    LOGGER.info(metrics.classification_report(y_eval_base, base_predictions, target_names=CLASSES))
    LOGGER.info(metrics.classification_report(y_eval_base, predictions, target_names=CLASSES))
    LOGGER.info("confusion matrix (deny):")
    LOGGER.info(metrics.confusion_matrix(y_eval_deny, deny_predictions))
    LOGGER.info("confusion matrix (query):")
    LOGGER.info(metrics.confusion_matrix(y_eval_query, query_predictions))
    LOGGER.info("confusion matrix (base):")
    LOGGER.info(metrics.confusion_matrix(y_eval_base, base_predictions))
    LOGGER.info("confusion matrix (combined):")
    LOGGER.info(metrics.confusion_matrix(y_eval_base, predictions))

    # Uncomment to see vocabulary
    # LOGGER.info(pipeline.get_params()['union__tweet_text__count'].get_feature_names())

    # Convert results to dict of tweet ID to predicted class
    results = {}
    for (i, prediction) in enumerate(predictions):
        results[tweets_eval[i]['id_str']] = prediction

    return results


def generate_one_vs_rest_annotations(annotations, one):
    """Convert annotation labels into a set of class vs not class.

    :param annotations:
        set of annotations for tweet IDs
    :type annotations
        `dict`
    :param one:
        the one annotation vs rest
    :type one:
        `str`
    """
    one_vs_rest_annotations = {}
    for tweet_id in annotations:
        one_vs_rest_annotations[tweet_id] = \
                annotations[tweet_id] if annotations[tweet_id] == one else 'not_{}'.format(one)
    return one_vs_rest_annotations


def build_query_pipeline():
    """Build a pipeline for predicting if a tweet is classified as query or not."""
    return Pipeline([
        # Extract useful features from tweets
        ('extract_tweets', TweetDetailExtractor(strip_hashtags=True, strip_mentions=True)),

        # Combine processing of features
        ('union', FeatureUnion(
            transformer_list=[

                ('count_depth', Pipeline([
                    ('selector', ItemSelector(keys='depth')),
                    ('count', FeatureCounter(names='depth')),
                    ('vect', DictVectorizer()),
                ])),

                ('is_news', Pipeline([
                    ('selector', ItemSelector(keys='is_news')),
                    ('count', FeatureCounter(names='is_news')),
                    ('vect', DictVectorizer()),
                ])),

                ('is_root', Pipeline([
                    ('selector', ItemSelector(keys='is_root')),
                    ('count', FeatureCounter(names='is_root')),
                    ('vect', DictVectorizer()),
                ])),

                # Count positive and negative words in the tweets
                ('pos_neg_sentiment', Pipeline([
                    ('selector', ItemSelector(keys=['positive_words', 'negative_words'])),
                    ('count', FeatureCounter(names=['positive_words', 'negative_words'])),
                    ('vect', DictVectorizer()),
                ])),

                # Count querying words in the tweets
                ('querying_words', Pipeline([
                    ('selector', ItemSelector(keys='querying_words')),
                    ('count', FeatureCounter(names='querying_words')),
                    ('vect', DictVectorizer()),
                ])),

                # Count number of question marks
                ('count_question_marks', Pipeline([
                    ('selector', ItemSelector(keys='question_mark_count')),
                    ('count', FeatureCounter(names='question_mark_count')),
                    ('vect', DictVectorizer()),
                ])),

            ],

            # Relative weights of transformations
            transformer_weights={
                'question_mark_count': 5.0,
                'count_depth': 1.0,
                'pos_neg_sentiment': 1.0,
                'querying_words': 5.0,
                'is_news': 2.5,
                'is_root': 2.5,
            },

        )),

        # Use a classifier on the result
        ('classifier', SVC(kernel='linear', class_weight='balanced'))

    ])


def build_deny_pipeline():
    """Build a pipeline for predicting if a tweet is classified as deny or not."""
    return Pipeline([
        # Extract useful features from tweets
        ('extract_tweets', TweetDetailExtractor(strip_hashtags=True, strip_mentions=True)),

        # Combine processing of features
        ('union', FeatureUnion(
            transformer_list=[

                ('count_ellipsis', Pipeline([
                    ('selector', ItemSelector(keys='ellipsis_count')),
                    ('count', FeatureCounter(names='ellipsis_count')),
                    ('vect', DictVectorizer()),
                ])),

                # Count number of question marks
                ('count_question_marks', Pipeline([
                    ('selector', ItemSelector(keys='question_mark_count')),
                    ('count', FeatureCounter(names='question_mark_count')),
                    ('vect', DictVectorizer()),
                ])),

                # Count occurrences on tweet text
                ('tweet_text', Pipeline([
                    ('selector', ItemSelector(keys='text_minus_root')),
                    ('list_to_str', pipelinize(list_to_str)),
                    ('count', TfidfVectorizer()),
                ])),

                ('count_depth', Pipeline([
                    ('selector', ItemSelector(keys='depth')),
                    ('count', FeatureCounter(names='depth')),
                    ('vect', DictVectorizer()),
                ])),

                ('is_news', Pipeline([
                    ('selector', ItemSelector(keys='is_news')),
                    ('count', FeatureCounter(names='is_news')),
                    ('vect', DictVectorizer()),
                ])),

                ('is_root', Pipeline([
                    ('selector', ItemSelector(keys='is_root')),
                    ('count', FeatureCounter(names='is_root')),
                    ('vect', DictVectorizer()),
                ])),

                # Count positive and negative words in the tweets
                ('pos_neg_sentiment', Pipeline([
                    ('selector', ItemSelector(keys=['positive_words', 'negative_words'])),
                    ('count', FeatureCounter(names=['positive_words', 'negative_words'])),
                    ('vect', DictVectorizer()),
                ])),

                # Count denying words in the tweets
                ('denying_words', Pipeline([
                    ('selector', ItemSelector(keys='denying_words')),
                    ('count', FeatureCounter(names='denying_words')),
                    ('vect', DictVectorizer()),
                ])),

                # Count querying words in the tweets
                ('querying_words', Pipeline([
                    ('selector', ItemSelector(keys='querying_words')),
                    ('count', FeatureCounter(names='querying_words')),
                    ('vect', DictVectorizer()),
                ])),

                # Count swear words and personal attacks
                ('offensiveness', Pipeline([
                    ('selector', ItemSelector(keys=['swear_words', 'personal_words'])),
                    ('count', FeatureCounter(names=['swear_words', 'personal_words'])),
                    ('vect', DictVectorizer()),
                ])),

            ],

            # Relative weights of transformations
            transformer_weights={
                # 'count_depth': 1.0,
                # 'pos_neg_sentiment': 1.0,
                # 'denying_words': 10.0,
                # 'offensiveness': 10.0,
                # 'is_news': 2.5,
                # 'is_root': 2.5,

                'count_ellipsis': 5.0,
                'tweet_text': 2.0,
                'count_depth': 1.0,
                'pos_neg_sentiment': 1.0,
                'denying_words': 5.0,
                'offensiveness': 10.0,
                'is_news': 2.5,
                'is_root': 2.5,

                'question_mark_count': 5.0,
                # 'count_depth': 1.0,
                # 'pos_neg_sentiment': 1.0,
                'querying_words': 5.0,
                # 'is_news': 2.5,
                # 'is_root': 2.5,
            },

        )),

        # Use a classifier on the result
        ('classifier', SVC(kernel='linear', class_weight='balanced'))

    ])


def build_base_pipeline():
    """Build a pipeline for predicting all 4 SDQC classes."""
    return Pipeline([
        # Extract useful features from tweets
        ('extract_tweets', TweetDetailExtractor(strip_hashtags=True, strip_mentions=True)),

        # Combine processing of features
        ('union', FeatureUnion(
            transformer_list=[

                # Count occurrences on tweet text
                ('tweet_text', Pipeline([
                    ('selector', ItemSelector(keys='text_stemmed_stopped')),
                    ('list_to_str', pipelinize(list_to_str)),
                    ('count', TfidfVectorizer()),
                ])),

                # Punctuation
                ('count_periods', Pipeline([
                    ('selector', ItemSelector(keys='period_count')),
                    ('count', FeatureCounter(names='period_count')),
                    ('vect', DictVectorizer()),
                ])),

                ('count_question_marks', Pipeline([
                    ('selector', ItemSelector(keys='question_mark_count')),
                    ('count', FeatureCounter(names='question_mark_count')),
                    ('vect', DictVectorizer()),
                ])),

                ('count_exclamations', Pipeline([
                    ('selector', ItemSelector(keys='exclamation_count')),
                    ('count', FeatureCounter(names='exclamation_count')),
                    ('vect', DictVectorizer()),
                ])),

                # ('count_ellipsis', Pipeline([
                #     ('selector', ItemSelector(keys='ellipsis_count')),
                #     ('count', FeatureCounter(names='ellipsis_count')),
                #     ('vect', DictVectorizer()),
                # ])),

                # Count features
                ('count_chars', Pipeline([
                    ('selector', ItemSelector(keys='char_count')),
                    ('count', FeatureCounter(names='char_count')),
                    ('vect', DictVectorizer()),
                ])),

                # ('count_depth', Pipeline([
                #     ('selector', ItemSelector(keys='depth')),
                #     ('count', FeatureCounter(names='depth')),
                #     ('vect', DictVectorizer()),
                # ])),

                ('count_hashtags', Pipeline([
                    ('selector', ItemSelector(keys='hashtags')),
                    ('count', FeatureCounter(names='hashtags')),
                    ('vect', DictVectorizer()),
                ])),

                ('count_mentions', Pipeline([
                    ('selector', ItemSelector(keys='user_mentions')),
                    ('count', FeatureCounter(names='user_mentions')),
                    ('vect', DictVectorizer()),
                ])),

                ('count_retweets', Pipeline([
                    ('selector', ItemSelector(keys='retweet_count')),
                    ('count', FeatureCounter(names='retweet_count')),
                    ('vect', DictVectorizer()),
                ])),

                # Boolean features
                ('is_news', Pipeline([
                    ('selector', ItemSelector(keys='is_news')),
                    ('count', FeatureCounter(names='is_news')),
                    ('vect', DictVectorizer()),
                ])),

                ('is_root', Pipeline([
                    ('selector', ItemSelector(keys='is_root')),
                    ('count', FeatureCounter(names='is_root')),
                    ('vect', DictVectorizer()),
                ])),

                ('verified', Pipeline([
                    ('selector', ItemSelector(keys='verified')),
                    ('count', FeatureCounter(names='verified')),
                    ('vect', DictVectorizer()),
                ])),

                # Count positive and negative words in the tweets
                ('pos_neg_sentiment', Pipeline([
                    ('selector', ItemSelector(keys=['positive_words', 'negative_words'])),
                    ('count', FeatureCounter(names=['positive_words', 'negative_words'])),
                    ('vect', DictVectorizer()),
                ])),

                # Count denying words in the tweets
                ('denying_words', Pipeline([
                    ('selector', ItemSelector(keys='denying_words')),
                    ('count', FeatureCounter(names='denying_words')),
                    ('vect', DictVectorizer()),
                ])),

                # Count querying words in the tweets
                ('querying_words', Pipeline([
                    ('selector', ItemSelector(keys='querying_words')),
                    ('count', FeatureCounter(names='querying_words')),
                    ('vect', DictVectorizer()),
                ])),

                # Count swear words and personal attacks
                ('offensiveness', Pipeline([
                    ('selector', ItemSelector(keys=['swear_words', 'personal_words'])),
                    ('count', FeatureCounter(names=['swear_words', 'personal_words'])),
                    ('vect', DictVectorizer()),
                ])),

            ],

            # Relative weights of transformations
            transformer_weights={
                'tweet_text': 1.0,
                'count_hashtags': 0.5,
                'count_mentions': 0.5,
                'count_retweets': 0.5,
                'count_periods': 0.25,
                'count_question_marks': 0.25,
                'count_exclamations': 0.25,
                'count_chars': 0.5,
                # 'count_depth': 0.5,
                'verified': 0.5,
                'pos_neg_sentiment': 1.0,
                'denying_words': 20.0,
                'querying_words': 10.0,
                'offensiveness': 20.0,
                'is_news': 10.0,
                'is_root': 20.0,
            },

        )),

        # Use a classifier on the result
        ('classifier', SVC(kernel='rbf'))

    ])