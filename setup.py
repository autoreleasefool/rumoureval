"""RumourEval Setup."""

from setuptools import setup

setup(name='rumoureval',
      version='0.1.0',
      packages=['rumoureval'],
      entry_points={
          'console_scripts': [
              'rumoureval = rumoureval.__main__:main'
          ]
      })
