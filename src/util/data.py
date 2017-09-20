"""
Manages importing data into basic Python structures.
"""

import logging
import json
import os
import sys

LOGGER = logging.getLogger()

def get_script_path():
    """
    Get the root path which the script was run from.
    :rtype:
        `str`
    """
    return os.path.dirname(os.path.realpath(sys.argv[0]))

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
    thread = {}

    with open(os.path.join(folder, 'structure.json')) as structure:
        thread['structure'] = json.load(structure)

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

    for child in os.listdir(os.path.join(folder, 'source-tweet')):
        with open(os.path.join(folder, 'source-tweet', child)) as source_tweet:
            thread['source'] = json.load(source_tweet)

    thread['replies'] = []
    for child in os.listdir(os.path.join(folder, 'replies')):
        with open(os.path.join(folder, 'replies', child)) as reply_tweet:
            thread['replies'].append(json.load(reply_tweet))

    if os.path.exists(os.path.join(folder, 'context', 'wikipedia')):
        with open(os.path.join(folder, 'context', 'wikipedia')) as wiki:
            thread['wiki'] = wiki.read()
    else:
        thread['context/wiki'] = None

    if os.path.exists(os.path.join(folder, 'context', 'urls')):
        thread['context/urls'] = {}
        for child in os.listdir(os.path.join(folder, 'context', 'urls')):
            with open(os.path.join(folder, 'context', 'urls', child)) as url_context:
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
        `list` of `dict` or None
    """
    if not os.path.exists(folder):
        LOGGER.debug('File/folder does not exist: {}'.format(folder))
        return None

    if os.path.isfile(folder):
        return None

    tweet_data = []
    children = os.listdir(folder)
    if 'structure.json' in children:
        # This is the root of a twitter thread
        tweet_data.append(import_thread(folder))
    else:
        tweet_data = [import_tweet_data(os.path.join(folder, child)) for child in children]
    return list(filter(None, tweet_data))


def import_annotation_data(folder):
    """
    Imports raw annotation data for the specified data source, indicating the annotation
    for each tweet ID
    :param folder:
        folder of annotations
    :type folder:
        `str`
    :rtype:
        `dict`
    """
    task_a_annotations = {}
    task_b_annotations = {}
    with open(os.path.join(folder, 'subtaskA.json')) as annotation_json:
        task_a_annotations = json.load(annotation_json)
    with open(os.path.join(folder, 'subtaskB.json')) as annotation_json:
        task_a_annotations = json.load(annotation_json)
    return task_a_annotations, task_b_annotations


def import_data(datasource):
    """
    Imports raw tweet data from the specified data source, to be parsed later.
    :param datasource:
        source of data to import
    :type datasource:
        either 'dev', 'train', or 'test'
    :rtype:
        `list` of `dict`, `dict`
    """
    source_folder = os.path.join(get_script_path(), '..', 'data', datasource)
    source_annotations = os.path.join(get_script_path(),
                                      '..', 'data', '{}-annotations'.format(datasource))
    tweet_data = import_tweet_data(source_folder)
    annotation_data = import_annotation_data(source_annotations)
    return tweet_data, annotation_data
