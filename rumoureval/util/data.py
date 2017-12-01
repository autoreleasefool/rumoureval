"""
Manages importing data into basic Python structures.
"""

import logging
import json
import os
import sys
from time import time
import magic
from .lists import filter_none
from .log import get_log_separator
from ..pipeline.tweet_detail_extractor import TweetDetailExtractor
from ..objects.tweet import Tweet


LOGGER = logging.getLogger()


def size_mb(docs):
    """
    Get the size of a list of docs in megabytes

    :param docs:
        the documents
    :type docs:
        `iterable` of `str`
    :rtype:
        `int`
    """
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6


def get_script_path():
    """
    Get the root path which the script was run from.

    :rtype:
        `str`
    """
    return os.path.dirname(os.path.realpath(sys.argv[0]))


def get_output_path():
    """
    Get the path to which output should be saved.

    :rtype:
        `str`
    """
    return os.path.join(get_script_path(), '..', 'output')


def get_datasource_path(datasource, annotations=False):
    """
    Get the path to input data.

    :param datasource:
        source of data to import
    :type datasource:
        either 'dev', 'train', or 'test'
    :param annotations:
        `True` for data annotations, `False` for data. Defaults to `False`
    :type annotations:
        `bool`
    :rtype:
        `str`
    """
    directory = datasource if not annotations else '{}-annotations'.format(datasource)
    return os.path.join(get_script_path(), '..', 'data', directory)


def import_thread(folder):
    """
    Imports a single twitter thread following a specific structure into a `dict`.
    Assumes the following structure:

    folder
    |- structure.json
    |- urls.dat
    |- source-tweet
        |- <tweet_id>.json
    |- replies
        |- <tweet_id>.json
        |- ...
    |- context
        |- wikipedia
        |- urls
            |- <url_md5>
            |- ...

    :param folder:
        folder name to retrieve source tweet from
    :type folder:
        `str`
    :rtype:
        `dict`
    """
    # pylint:disable=too-many-branches
    thread = {}

    if os.path.exists(os.path.join(folder, 'structure.json')):
        with open(os.path.join(folder, 'structure.json')) as structure:
            thread['structure'] = json.load(structure)

    if os.path.exists(os.path.join(folder, 'urls.dat')):
        with open(os.path.join(folder, 'urls.dat')) as url_dat:
            thread['urls'] = []
            for line in url_dat.readlines():
                raw_url = line.split()
                url = {
                    'hash': raw_url[0],
                    'short': raw_url[1],
                    'full': raw_url[2],
                }
                thread['urls'].append(url)

    if os.path.exists(os.path.join(folder, 'source-tweet')):
        for child in os.listdir(os.path.join(folder, 'source-tweet')):
            with open(os.path.join(folder, 'source-tweet', child)) as source_tweet:
                thread['source'] = json.load(source_tweet)

    thread['replies'] = {}
    if os.path.exists(os.path.join(folder, 'replies')):
        for child in os.listdir(os.path.join(folder, 'replies')):
            with open(os.path.join(folder, 'replies', child)) as reply_tweet_file:
                reply_tweet = json.load(reply_tweet_file)
                thread['replies'][reply_tweet['id_str']] = reply_tweet

    if os.path.exists(os.path.join(folder, 'context', 'wikipedia')):
        with open(os.path.join(folder, 'context', 'wikipedia')) as wiki:
            thread['wiki'] = wiki.read()
    else:
        thread['context/wiki'] = None

    if os.path.exists(os.path.join(folder, 'context', 'urls')):
        thread['context/urls'] = {}
        for child in os.listdir(os.path.join(folder, 'context', 'urls')):
            context_file_path = os.path.join(folder, 'context', 'urls', child)
            if magic.from_file(context_file_path, mime=True) != 'text/html':
                continue
            with open(context_file_path) as url_context:
                thread['context/urls'][child] = url_context.read()
    else:
        thread['context/urls'] = None

    if os.path.exists(os.path.join(folder, 'urls-content')):
        thread['urls-content'] = {}
        for child in os.listdir(os.path.join(folder, 'urls-content')):
            with open(os.path.join(folder, 'urls-content', child)) as url_content:
                thread['urls-content'][child] = url_content.read()
    else:
        thread['urls-content'] = None

    return thread


def import_tweet_data(folder):
    """
    Imports raw tweet data from the given folder, recursively.

    :param folder:
        folder name to retrieve tweets from
    :type folder:
        `str`
    :rtype:
        `dict` or None
    """
    if not os.path.exists(folder):
        LOGGER.debug('File/folder does not exist: %s', folder)
        return None

    if os.path.isfile(folder):
        return None

    tweet_data = []
    children = os.listdir(folder)
    if 'structure.json' in children:
        # This is the root of a twitter thread
        tweet_data.append(import_thread(folder))
    else:
        for child in children:
            child_data = import_tweet_data(os.path.join(folder, child))
            if child_data is not None:
                tweet_data += child_data

    tweet_data = filter_none(tweet_data)
    return tweet_data


def import_annotation_data(datasource):
    """
    Imports raw annotation data for the specified data source, indicating the annotation
    for each tweet ID

    :param datasource:
        source of data to import
    :type datasource:
        either 'dev', 'train', or 'test'
    :rtype:
        `dict`
    """
    task_a_annotations = {}
    task_b_annotations = {}
    with open(os.path.join(get_datasource_path(datasource, annotations=True),
                           'subtaskA.json')) as annotation_json:
        task_a_annotations = json.load(annotation_json)
    with open(os.path.join(get_datasource_path(datasource, annotations=True),
                           'subtaskB.json')) as annotation_json:
        task_b_annotations = json.load(annotation_json)
    return task_a_annotations, task_b_annotations


def build_tweet(tweet_data, tweet_id, structure, is_source=False):
    """
    Parses raw twitter data and creates Tweet objects, setting up their parent and child
    tweets according to the tweet_data structure.

    :param tweet_data:
        A single Twitter thread
    :type tweet_data:
        `dict`
    :param tweet_id:
        ID of tweet to build
    :type root_tweet_id:
        `str`
    :param structure:
        Structure that children tweets follow
    :type structure:
        `list` or `dict`
    :param is_source:
        True if the tweet is the source tweet of a thread, False otherwise
    :type is_source:
        `bool`
    :rtype:
        :class:`Tweet`
    """
    children = [
        build_tweet(
            tweet_data,
            child_tweet_id,
            structure[child_tweet_id]
        ) for child_tweet_id in structure
    ]
    children = filter_none(children)
    if is_source:
        return Tweet(tweet_data['source'], children=children, is_source=True)
    elif tweet_id in tweet_data['replies']:
        return Tweet(tweet_data['replies'][tweet_id], children=children)

    return None


def import_data(datasource):
    """
    Imports raw tweet data from the specified data source, to be parsed later.

    :param datasource:
        source of data to import
    :type datasource:
        either 'dev', 'train', or 'test'
    :rtype:
        `list` of :class:`Tweet`
    """
    LOGGER.info(get_log_separator())
    LOGGER.info('Beginning `%s` data import', datasource)
    start_time = time()
    source_folder = get_datasource_path(datasource)
    tweet_data = import_tweet_data(source_folder)

    parsed_tweets = [
        build_tweet(
            thread,
            thread['source']['id_str'],
            thread['structure'][thread['source']['id_str']],
            is_source=True
        ) for thread in tweet_data
    ]

    root_tweets = parsed_tweets[:]
    for tweet in parsed_tweets:
        parsed_tweets += list(tweet.children())

    LOGGER.info('Took %0.3fs to import `%s` data', time() - start_time, datasource)
    LOGGER.debug('Imported %d root tweets from %s', len(root_tweets), datasource)
    LOGGER.debug('Imported %d child tweets from %s', len(parsed_tweets), datasource)

    return parsed_tweets


def output_data_by_class(tweets, annotations, task, prefix=None):
    """Output files for each annotation, containing all the tweets similarly annotated.

    :param tweets:
        the list of tweets
    :type tweets:
        `list` of :class:`Tweet`
    :param annotations
        tweet IDs mapped to their annotations
    :type annotations:
        `dict`
    :param task:
        the task being output
    :type task:
        'A' or 'B'
    :param prefix:
        prefix for filenames
    :type prefix:
        `str` or None
    """
    detail_extractor = TweetDetailExtractor(task, strip_hashtags=False, strip_mentions=False)
    sorted_tweets = {}
    sorted_tweet_text = {}
    for tweet in tweets:
        annotation = annotations[tweet['id_str']]
        if annotation not in sorted_tweets:
            sorted_tweets[annotation] = []
            sorted_tweet_text[annotation] = set()
        sorted_tweets[annotation].append(tweet.raw())
        sorted_tweet_text[annotation] |= set(list(detail_extractor._tokenize(TweetDetailExtractor.get_parseable_tweet_text(tweet, task=task))))

    os.makedirs(get_output_path(), exist_ok=True)
    for annotation in sorted_tweets:
        filename = ('{0}_{1}.json' if prefix is not None else '{1}.json').format(prefix, annotation)
        with open(os.path.join(get_output_path(), filename), 'w') as file:
            json.dump(sorted_tweets[annotation], file, sort_keys=True, indent=2)
        if task == 'B':
            filename = '{0}_dict.txt'.format(annotation)
            with open(os.path.join(get_output_path(), filename), 'w') as file:
                file.write('\n'.join(sorted(list(sorted_tweet_text[annotation]))))


