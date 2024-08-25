import click
from pathlib import Path
import pickle
from md_sfa.main import TrajProcessor
import numpy as np
import pandas as pd

# Define helper functions for saving, loading, and deleting the TrajProcessor instance
def save_processor_instance(processor_instance, file_path='processor_instance.pkl'):
    with open(file_path, 'wb') as file:
        pickle.dump(processor_instance, file)

def load_processor_instance(file_path='processor_instance.pkl'):
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        return TrajProcessor()

def delete_processor_instance(file_path='processor_instance.pkl'):
    Path(file_path).unlink(missing_ok=True)

@click.group()
def cli():
    pass

@cli.command()
@click.option('--path_to_trajectories', type=click.Path(), help='Path to the trajectories directory.')
@click.option('--topology_file', type=click.Path(), help='Path to the topology file.')
@click.option('--stride', default=1, help='Stride for loading trajectories.')
@click.option('--atom_indices', help='Atom indices for selection.', default=None)
def load_trajectories(path_to_trajectories, topology_file, stride, atom_indices):
    processor_instance = load_processor_instance()
    processor_instance.load_trajectories(Path(path_to_trajectories), Path(topology_file), stride, atom_indices)
    save_processor_instance(processor_instance)

@cli.command()
@click.option('--types', multiple=True, help='Types of dihedrals to featurize.')
@click.option('--nosincos', is_flag=True, default=False, help='Disable sin/cos transformation.')
def featurize(types, nosincos):
    processor_instance = load_processor_instance()
    sincos = not nosincos
    processor_instance.featurize(list(types), sincos)
    save_processor_instance(processor_instance)

@cli.command()
@click.option('--nosincos', is_flag=True, default=False, help='Disable sin/cos transformation.')
def describe_features(nosincos):
    processor_instance = load_processor_instance()
    sincos = not nosincos
    processor_instance.describe_features(sincos)
    save_processor_instance(processor_instance)

@cli.command()
@click.option('--description_file_path', type=click.Path(), help='Path to save the description file.')
@click.option('--nosincos', is_flag=True, default=False, help='Disable sin/cos transformation.')
def dump_description(description_file_path, nosincos):
    processor_instance = load_processor_instance()
    sincos = not nosincos
    processor_instance.dump_description(description_file_path, sincos)
    save_processor_instance(processor_instance)

@cli.command()
@click.option('--dump_file_path', type=click.Path(), help='Path to save the featurized data.')
@click.option('--nosincos', is_flag=True, default=False, help='Disable sin/cos transformation.')
def dump_featurized(dump_file_path, nosincos):
    processor_instance = load_processor_instance()
    sincos = not nosincos
    processor_instance.dump_featurized(dump_file_path, sincos)
    save_processor_instance(processor_instance)

@cli.command()
@click.option('--n_components', default=2, help='Number of components for SFA.')
@click.option('--tau', default=1, help='Tau parameter for SFA.')
def run_sfa(n_components, tau):
    processor_instance = load_processor_instance()
    processor_instance.run_sfa(n_components, tau)
    save_processor_instance(processor_instance)

@cli.command()
@click.option('--plumed_filename', type=click.Path(), help='Path to save the PLUMED file.')
def create_plumed_file(plumed_filename):
    processor_instance = load_processor_instance()
    processor_instance.create_plumed_file(plumed_filename)
    save_processor_instance(processor_instance)

@cli.command()
def restart():
    delete_processor_instance()

@cli.command()
@click.option('--save_file', type=click.Path(), help='Path to save the SFA components.')
def dump_sfa_components(save_file):
    processor_instance = load_processor_instance()
    processor_instance.dump_sfa_components(save_file)
    save_processor_instance(processor_instance)

@cli.command()
@click.option('--dat_file', type=click.Path(exists=True))
@click.option('--pdb_input', type=click.Path(exists=True))
@click.option('--pdb_output', type=click.Path())
def plumed_bfactor(dat_file, pdb_input, pdb_output):
    """Calculate B-factors using PLUMED input and PDB files."""
    processor = TrajProcessor()
    processor.process_bfactor(dat_file, pdb_input, pdb_output)

if __name__ == '__main__':
    cli()
