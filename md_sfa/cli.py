import click
from pathlib import Path
import pickle
from md_sfa.main import TrajProcessor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from timelaggedcv import TimeLaggedCV

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
@click.option('--algorithm', default='kmeans', help='Clustering algorithm to use.')
@click.option('--n_clusters', default=100, help='Number of clusters for algorithm (if applicable).')
def cluster(algorithm, n_clusters):
    processor_instance = load_processor_instance()
    if algorithm == 'GMM':
        processor_instance.cluster(algorithm)
    else:
        processor_instance.cluster(algorithm, n_clusters)
    save_processor_instance(processor_instance)

@cli.command()
@click.option('--num_samples', default=3, help='Number of samples to draw from clustering.')
def dump_clusters(num_samples):
    processor_instance = load_processor_instance()
    processor_instance.dump_clusters(num_samples)
    save_processor_instance(processor_instance)

@cli.command()
@click.option('--ensemble_one', type=click.Path(), help='Path to the first ensemble file.')
@click.option('--ensemble_two', type=click.Path(), help='Path to the second ensemble file.')
@click.option('--ensemble_features', type=click.Path(), help='Path to the ensemble features file.')
def classify(ensemble_one, ensemble_two, ensemble_features):
    processor_instance = load_processor_instance()
    processor_instance.classify(ensemble_one, ensemble_two, ensemble_features)
    save_processor_instance(processor_instance)

@cli.command()
@click.option('--plumed_filename', type=click.Path(), help='Output path of the PLUMED file.')
def create_classifier_plumed(plumed_filename):
    processor_instance = load_processor_instance()
    processor_instance.classifier_plumed(plumed_filename)
    save_processor_instance(processor_instance)

@cli.command()
@click.option('--pickle_descriptor', multiple=True, default=['.','.'], help='One or more file paths for the pickle descriptor (e.g., file1.pkl file2.pkl ...)')
@click.option('--pickle_features', multiple=True, default=['.','.'],help='One or more file paths for the pickle features (e.g., file1.pkl file2.pkl ...)')
@click.option('--tau', type=int, default=10, show_default=True, help='Time lag parameter (integer)')
def train_vae(pickle_descriptor, pickle_features, tau):
    options = TimeLaggedCV.DEFAULT
    options['create_dataset_options']['pickle_descriptor'] = list(pickle_descriptor)
    options['create_dataset_options']['pickle_features'] = list(pickle_features)
    options['dataset']['tau'] = tau
    model = TimeLaggedCV(options)
    model.run()

@cli.command()
@click.option(
    '--pickle_descriptor',
    multiple=True,
    required=True,
    help='Space-separated list of paths for pickle_descriptor'
)
@click.option(
    '--pickle_features',
    multiple=True,
    required=True,
    help='Space-separated list of paths for pickle_features'
)
@click.option(
    '--tau',
    type=int,
    required=True,
    help='Integer value for tau'
)
@click.option(
    '--model_path',
    required=True,
    help='Path to model file (e.g. model_final.pt or model.pt)'
)
@click.option(
    '--teacher_flag',
    is_flag=True,
    help='Set this flag if teacher mode is desired (default is False)'
)
def predict_vae(pickle_descriptor, pickle_features, tau, model_path, teacher_flag):
    options = TimeLaggedCV.DEFAULT
    options['create_dataset_options']['pickle_descriptor'] = list(pickle_descriptor)
    options['create_dataset_options']['pickle_features'] = list(pickle_features)
    options['dataset']['tau'] = tau
    options['general']['teacher'] = teacher_flag

    tlcv = TimeLaggedCV(options)
    model = tlcv.get_model()

    if not teacher_flag:
        model.load(model_path)
    else:
        model.load(model_path)

    cv_space = model.estimate_cv_from_dataset(tlcv.dataset_weight, batchsize=100000)
    list_cv = np.split(cv_space, tlcv.dataset_weight.total_length)

    if not teacher_flag:
        np.savez('cv_pred.npz', *list_cv)
        print('Saved teacher prediction')
    else:
        np.savez('cv_pred_student.npz', *list_cv)
        print('Saved student prediction')

    skip = 1
    for start, end in zip(tlcv.dataset_weight.total_length[:-1], tlcv.dataset_weight.total_length[1:]):
        plt.plot(*cv_space[start:end:skip].T, '.')
    plt.show()

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
    processor_instance = load_processor_instance()
    processor_instance.process_bfactor(dat_file, pdb_input, pdb_output)
    save_processor_instance(processor_instance)

if __name__ == '__main__':
    cli()
