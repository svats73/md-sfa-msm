#! /usr/bin/env python

"""
This package has been based on the fantastic scikit-learn-contrib project skeleton and its documentation,
as provided by Vighnesh Birodkar and scikit-learn-contrib contributors.
It can be found https://github.com/scikit-learn-contrib/project-template

Thanks a lot, you made our work significantly easier!
"""

import codecs
import os

from setuptools import find_packages, setup

# get __version__ from _version.py
ver_file = os.path.join('sksfa', '_version.py')
with open(ver_file) as f:
    exec(f.read())

DISTNAME = 'sklearn-sfa'
DESCRIPTION = 'A scikit-learn compatible implementation of Slow Feature Analysis'
with codecs.open('README.rst', encoding='utf-8-sig') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'Merlin Schüler'
MAINTAINER_EMAIL = 'merlin.schueler@ini.ruhr-uni-bochum.de'
URL = 'https://github.com/wiskott-lab/sklearn-sfa'
LICENSE = 'new BSD'
DOWNLOAD_URL = 'https://github.com/wiskott-lab/sklearn-sfa'
VERSION = __version__
INSTALL_REQUIRES = ['numpy', 'scipy', 'scikit-learn']
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
      #         'Programming Language :: Python :: 2.7',
      #         'Programming Language :: Python :: 3.5',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7']
EXTRAS_REQUIRE = {
    'tests': [
        'pytest',
        'pytest-cov'],
    'docs': [
        'sphinx',
        'sphinx-gallery',
        'sphinx_rtd_theme',
        'numpydoc',
        'matplotlib'
    ]
}

setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
      zip_safe=False,  # the package can run out of an .egg file
      classifiers=CLASSIFIERS,
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE)
