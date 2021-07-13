#!/usr/bin/env python
# -*- coding: utf-8 -*-

from distutils.core import setup

NAME = 'sampleproject'
VERSION = '0.0.1'
DESCRIPTION = 'A sample Python project.'
URL = 'https://github.com/me/myproject'
AUTHOR = 'A. Random Developer'
AUTHOR_EMAIL = 'author@example.com'
MAINTAINER = AUTHOR
MAINTAINER_EMAIL = AUTHOR_EMAIL
LICENSE = 'MIT'
# LONG_DESCRIPTION = ''

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    # long_description=LONG_DESCRIPTION,
    # long_description_content_type='text/markdown',
    url=URL,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    license = LICENSE,
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

    package_dir={'genetic': 'src'},

    packages=['genetic'], 

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