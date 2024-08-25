from setuptools import setup, find_packages

setup(
    name='md-sfa',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'click',
        'numpy',
        'pandas',
        'pickle5',
        'mdtraj',
        'biopython',
    ],
    entry_points={
        'console_scripts': [
            'md-sfa=md_sfa.cli:cli',
        ],
    },
)
