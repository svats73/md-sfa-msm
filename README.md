# MD-SFA: Molecular Dynamics Slow Feature Analysis CLI Tool

## Overview

MD-SFA is a powerful command-line interface (CLI) tool designed for the analysis of molecular dynamics (MD) simulations. Leveraging the capabilities of the `md_sfa` library, MD-SFA facilitates the loading of MD trajectory data, the featurization of trajectories, the execution of Slow Feature Analysis (SFA), and the creation of PLUMED files for biasing simulations based on SFA components. 

## Custom Installation

Before installing MD-SFA, it is necessary to install a custom version of `sklearnsfa`, which is packaged within the `md-sfa` repository. This ensures compatibility and optimal performance for SFA computations within MD-SFA. Follow these steps to install both the custom `sklearnsfa` and `md-sfa`:

1. **Clone the `md-sfa` repository to your local machine:**
``` git clone https://github.com/svats73/md-sfa-msm.git ```

2. **Navigate to the cloned repository directory:**
``` cd md-sfa-msm ```

3. **Install the custom `sklearnsfa` package:**
``` pip install ./sklearn-sfa ```

4. **Install the `md-sfa` package:**
``` pip install . ```

This installation process will ensure that you have both the `md-sfa` tool and the custom `sklearn-sfa` library installed and ready for your MD analysis tasks.

Since the custom `sklearn-sfa` is installed directly from the `md-sfa` repository, it does not need to be listed in `requirements.txt`. However, ensure all other dependencies are installed by running:

``` pip install -r requirements.txt ```

## Usage

The MD-SFA CLI tool supports various commands for processing and analyzing your MD trajectories. Below is a guide to using these commands:

### Loading Trajectories

``` md-sfa load_trajectories --path_to_trajectories PATH --topology_file FILE --stride N --atom_indices "selection" ```

- `--path_to_trajectories`: Directory containing trajectory files.
- `--topology_file`: Topology file path.
- `--stride`: Interval for loading frames (optional).
- `--atom_indices`: Atom selection string (optional).

### Featurizing Dihedrals

``` md-sfa featurize --types TYPE1 TYPE2 --nosincos ```

- `--types`: Types of dihedrals to featurize. Can specify multiple types.
- `--nosincos`: Disables the sin/cos transformation if set.

### Describing Features

``` md-sfa describe_features --nosincos ```

- `--nosincos`: Disables the sin/cos transformation if set.

### Dumping Description

``` md-sfa dump_description --description_file_path PATH --nosincos ```

- `--description_file_path`: File path to save the feature description.
- `--nosincos`: Dump non-transformed feature description if created.

### Dumping Featurized Data

``` md-sfa dump_featurized --dump_file_path PATH --nosincos ```

- `--dump_file_path`: File path to save the featurized data.
- `--nosincos`: Dump non-transformed features if created.

### Running Slow Feature Analysis (SFA)

``` md-sfa run_sfa --n_components N --tau T ```

- `--n_components`: Number of SFA components to extract.
- `--tau`: The tau parameter for SFA.

### Creating PLUMED File

``` md-sfa create_plumed_file --plumed_filename FILENAME ```

- `--plumed_filename`: File path to save the generated PLUMED file.

### Dumping SFA Components

``` md-sfa dump_sfa_components --save_file FILE ```

- `--save_file`: File path to save the SFA components.

### Restarting the Tool

To clear the current state and start fresh:

``` md-sfa restart ```

This command deletes any serialized state, allowing you to start a new analysis without interference from previous runs.
