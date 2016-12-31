#   Copyright 2016 The Trustees of Indiana University.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

""" Setuptools script """

import os
import argparse

from os.path import join, exists, dirname, abspath, expanduser
from setuptools import setup

_wd = abspath(os.curdir)
base = abspath(dirname(__file__))

# change to the project directory
os.chdir(base)

kwargs = dict(
    name="relsifter",
    description='Triple scoring task (WSDM Cup 2017)',
    version='0.1.0',
    author='Prashant Shiralkar and others (see CONTRIBUTORS.md)',
    author_email='pshiralk@indiana.edu',
    packages=[
        'relsifter', 'relsifter.activity', 'relsifter.textprofile', 'relsifter.model',
        'relsifter.tests', 'relsifter.datastructures'
    ],
    include_package_data=True,
    install_requires=[
        'numpy',
        'scipy',
    ],
    entry_points={
        'console_scripts': [
            'relsifter = relsifter.__main__:main'
        ]
    }
)

parser = argparse.ArgumentParser(description=__file__, add_help=False)

if __name__ == '__main__':
    args, rest = parser.parse_known_args()
    kwargs['script_args'] = rest
    setup(**kwargs)

# back to the working directory
os.chdir(_wd)
