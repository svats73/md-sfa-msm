from setuptools import setup, find_packages

setup(
    name='md-sfa',
    version='0.2.0',
    packages=find_packages(),
    install_requires=[
        'click',
        'numpy',
        'pandas',
        #'pickle5',
        'mdtraj',
        'biopython',
        'scikit-learn',
    ],
    entry_points={
        'console_scripts': [
            'md-sfa=md_sfa.cli:cli',
        ],
    },
)
