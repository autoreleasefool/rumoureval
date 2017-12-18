# CSI4900

[RumourEval](http://alt.qcri.org/semeval2017/task8/): Determining rumour veracity and support for rumours

## How to Run

### Dependencies

- Python3 (we recommend install with [pyenv](https://github.com/pyenv/pyenv))
- python-magic ([see dependencies](https://github.com/ahupp/python-magic#dependencies))
    - For Mac, `brew install libmagic`

### Running the code

`python3 -m rumoureval [--verbose] [--test] [--osorted] [--disable-cache] [--plot] [--trump]`

- `--verbose` to get verbose output
- `--test` to train on training data, then evaluate on test data. Without, model is tested on validation data
- `--osorted` to output tweets sorted into their classes for task A and B
- `--disable-cache` to force the task A classifier to retrain on training data. Used to speed up iterations on classifier in task B
- `--plot` to plot the confusion matrices of task A and B
- `--trump` to test classification of Trump tweets picked and labelled by ourselves

## Contributing

Ensure all code passes pylint and pycodestyle tests, with the following invocations:

- `pylint rumoureval setup.py`
- `pycodestyle --max-line-length=100 rumoureval setup.py`
