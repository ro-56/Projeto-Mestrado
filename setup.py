#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # Package meta-data.
NAME = 'sampleproject'
VERSION = '0.0.1'
DESCRIPTION = 'A sample Python project.'
URL = 'https://github.com/me/myproject'
AUTHOR = 'A. Random Developer'
EMAIL = 'author@example.com'


from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
LONG_DESCRIPTION = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    url=URL,
    author=AUTHOR,
    author_email=EMAIL,
    classifiers=[
        'Development Status :: 1 - Planning',

        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
    ],

    package_dir={'': 'src'},

    packages=find_packages(where='src'), 

    python_requires='>=3.9, <4',

    entry_points={
        'console_scripts': [
            'genetic=genetic:main',
        ],
    },
    # project_urls={ 
    #     'Bug Reports': 'https://github.com/pypa/sampleproject/issues',
    #     'Funding': 'https://donate.pypi.org',
    #     'Say Thanks!': 'http://saythanks.io/to/example',
    #     'Source': 'https://github.com/pypa/sampleproject/',
    # },
)