"""Scores the results of a RumourEval implementation. Uses the provided scoring script."""
# pylint:disable=too-few-public-methods

import logging
import json
import os.path
import re
import subprocess
from ..util.data import get_datasource_path, get_output_path, get_script_path
from ..util.log import get_log_separator


LOGGER = logging.getLogger()
_SCORER_PATH = os.path.join(get_script_path(), '..', 'scorer')
_DEBUG_REGEX = re.compile(r'^((un)?match(ed|ing)|\d+( matched)? entries)')


class Scorer(object):
    """
    Accepts the results of a RumourEval implementation, runs scoring scripts,
    and outputs results.
    """

    def __init__(self, task, datasource):
        """Initialize Scorer.

        :param task:
            task to score, 'A' or 'B'
        :type task:
            `str`
        """
        if task not in ['A', 'B']:
            raise ValueError('task must be A or B')
        self._task = task
        self._output_file = os.path.join(get_output_path(),
                                         'subtask{}Results.json'.format(self._task))
        self._annotation_file = os.path.join(get_datasource_path(datasource, annotations=True),
                                             'subtask{}.json'.format(task))

    def _export_results(self, results):
        """Export task results to the output directory.

        :param results:
            Results of task
        :type results:
            `dict`
        """
        os.makedirs(self._output_file[:self._output_file.rfind(os.sep)], exist_ok=True)
        with open(self._output_file, 'w') as file:
            json.dump(results, file, sort_keys=True, indent=2)

    def _clean_up(self):
        """Clean up results."""
        pass

    def score(self, results):
        """Scores the results of a task.

        :param results:
            Results of task
        :type results:
            `dict`
        """
        LOGGER.info(get_log_separator())
        LOGGER.info('Scoring results of task %s:', self._task)

        self._export_results(results)
        out = subprocess.run(
            [
                'python',
                os.path.join(_SCORER_PATH, 'scorer{}.py'.format(self._task)),
                self._annotation_file,
                self._output_file
            ],
            stdout=subprocess.PIPE).stdout.decode('utf8')

        out_lines = out.split('\n')

        LOGGER.info(get_log_separator(thick=False))
        LOGGER.info('Output from Scorer%s.py script:', self._task)
        LOGGER.debug(out_lines[0])
        for line in out_lines[1:]:
            if _DEBUG_REGEX.match(line):
                LOGGER.debug(line)
            elif line:
                LOGGER.info(line)

        self._clean_up()
