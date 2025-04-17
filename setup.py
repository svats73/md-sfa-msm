from setuptools import setup, find_packages

setup(
    name='mdml',
    version='0.3.0',
    packages=find_packages(),
    install_requires=[
        'click',
        'numpy<2',
        'pandas',
        #'pickle5',
        'addict',
        'h5py',
        'torch',
        'lightning',
        'ruamel.yaml',
        'matplotlib',
        'parmed',
        'mdtraj',
        'biopython',
        'scikit-learn',
    ],
    entry_points={
        'console_scripts': [
            'mdml=mdml.cli:cli',
        ],
    },
)
