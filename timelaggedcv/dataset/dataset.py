import torch

from timelaggedcv import load_yaml, dump_yaml
import numpy as np
import pandas as pd
from .utils import (
    set_feature_mean_std,
    find_pairs,
)

from torch.utils.data import Dataset
from pathlib import Path

class CV_timelagged(Dataset):
    """Dataset for creating features for training a time-lagged autoencoder to find optimal CVs.
    Parameters
    ----------
    feature_dir : str
        Feature directory.
    tau : int
        Time lag for the time-lagged autoencoder.
    """

    def __init__(self, tau, bias_mask=None) -> None:
        super().__init__()
        self.config = load_yaml("config.yaml")
        self.tau = tau
        self.bias_mask = bias_mask
        # Load features
        (
            self.features,
            self.metad_weights,
            self.times,
            self.splits,
            self.traj_to_workdir,
        ) = get_features_pickle(self.config)
        # Find the data pairs
        self.data_pairs = []
        self.weights_time = []
        for time in self.times:
            time = time - time[0]  # Let the time start at 0
            pairs, weights = find_pairs(time, self.tau)
            self.weights_time.append(weights)
            self.data_pairs.append(pairs)
        print("Loaded everything into memory.")
        self.n_trajs = len(self.data_pairs)
        len_trajs = np.zeros(self.n_trajs)
        for i, pairs_i in enumerate(self.weights_time):
            len_trajs[i] = len(pairs_i)
        self.total_length = np.cumsum(len_trajs).astype("int")

        (
            self.feat_nn_list,
            self.feat_indexes_input,
            self.input_dims,
            self.feat_indexes_output,
            self.output_dim,
            self.means,
            self.stds,
        ) = set_feature_mean_std(self.features, self.config)
        self.decoder_dict = self.config["decoder_dict"]

    def __len__(self):
        return int(self.total_length[-1])

    def find_traj_frame(self, idx):
        # first find traj number
        traj_n = self.n_trajs - (idx < self.total_length).sum()
        # find idx within the whole trajectory
        if traj_n > 0:
            idx = idx - self.total_length[traj_n - 1]
        return traj_n, int(idx)

    def __getitem__(self, idx):
        # first find traj number
        traj_n, idx_t = self.find_traj_frame(idx)

        idx_final = self.data_pairs[traj_n][0][idx_t]
        idx_tau_final = self.data_pairs[traj_n][1][idx_t]
        ret = []
        for feat_index in self.feat_indexes_input:
            ret.append(torch.Tensor(self.features[traj_n][idx_final][feat_index]))
        ret.append(
            torch.Tensor(self.features[traj_n][idx_tau_final][self.feat_indexes_output])
        )
        ret.append(torch.Tensor(self.weights_time[traj_n][idx_t]))

        return ret

class CVfind(Dataset):
    "Need this dataset to get only all frames with with optional weights for reweighting."

    def __init__(self, dataset: CV_timelagged, weights_flag=False) -> None:
        super().__init__()
        self.dataset = dataset
        len_trajs = np.zeros(len(dataset.features))
        for i, data in enumerate(dataset.features):
            len_trajs[i] = data.shape[0]
        self.total_length = np.cumsum(len_trajs).astype("int")
        self.weights_flag = weights_flag

    def __len__(self):
        return int(self.total_length[-1])

    def find_traj_frame(self, idx):
        # first find traj number
        traj_n = self.dataset.n_trajs - (idx < self.total_length).sum()
        # find idx within the whole trajectory
        if traj_n > 0:
            idx = idx - self.total_length[traj_n - 1]
        return traj_n, int(idx)

    def find_index_from_traj(self, workdir, frame):
        idx_new = frame
        for i in range(len(self.dataset.traj_to_workdir)):
            if self.dataset.traj_to_workdir[i] == workdir:
                if idx_new < self.total_length[i]:
                    return [
                        torch.Tensor(self.dataset.features[i][idx_new][feat_index])
                        for feat_index in self.dataset.feat_indexes_input
                    ], torch.Tensor(
                        self.dataset.features[i][idx_new][
                            self.dataset.feat_indexes_output
                        ]
                    )
                else:
                    idx_new = idx_new - self.total_length[i]
        raise ValueError("Could not find index in dataset")

    def __getitem__(self, idx):
        # first find traj number
        traj_n, idx_t = self.find_traj_frame(idx)
        ret = []
        for feat_index in self.dataset.feat_indexes_input:
            ret.append(torch.Tensor(self.dataset.features[traj_n][idx_t][feat_index]))
        if self.weights_flag:
            ret.append(torch.Tensor(self.dataset.metad_weights[traj_n][idx_t]))
        return ret

def create_config_pickle(pickle_descriptors_list, pickle_features_list, skip=1):

    dih_list_dec = []
    list_dih_idx = []
    feature_dict_list = []
    counter_feat = 0
    for pickle_descriptors in pickle_descriptors_list:
        data = pd.read_pickle(pickle_descriptors)
        feat_dih_list = []
        dih_list = []
    
        for idx in range(len(data)):
            feat_dih_list.append(
                {
                    "feature": "dihedrals",
                    "transform": data.iloc[idx]["otherinfo"],
                    "atom_mask": data.iloc[idx]['atominds'].tolist(),
                    "residues": data.iloc[idx]['resids']
                }
            )
            list_dih_idx.append(data.iloc[idx]['atominds'].tolist())
            dih_list.append(data.iloc[idx]['atominds'])
        dih_list = np.unique(np.concatenate(dih_list))
        dih_list_dec = np.concatenate([dih_list_dec, dih_list])
        
        feature_dict_i = {}
        feature_dict_i["dihedrals"] = {}
        feature_dict_i["dihedrals"]["features"] = feat_dih_list
        feature_dict_i["dihedrals"]["dihedral_list"] = dih_list
        feature_dict_i["cv_feature_index"] = np.arange(counter_feat, counter_feat+len(data))
        feature_dict_list.append(feature_dict_i)
        counter_feat += len(data)
    decoder_dict = {}
    decoder_dict_dihedral = {"type": "dihedrals"}
    decoder_dict_dihedral["indexes"]= list_dih_idx
    decoder_dict["features"] = [
        decoder_dict_dihedral,
    ]
    decoder_dict["n_atoms"] = len(dih_list_dec)
    decoder_dict["cv_feature_index"] = np.arange(counter_feat)
    decoder_dict["atoms_decoder"] = dih_list_dec
    all_feature_dict={}
    all_feature_dict["dihedrals"] = list_dih_idx
    all_feature_dict["features"] = feature_dict_i

    # create yaml file
    save_dict = {}
    save_dict["CV_feature_dicts"] = feature_dict_list
    save_dict["decoder_dict"] = decoder_dict
    save_dict["all_input_feature_dict"] = all_feature_dict
    save_dict["weights_flag"] = False
    save_dict["time_mode"] = 'c_t'
    save_dict["max_time"] = 100000
    save_dict['pickle_descriptors'] = pickle_descriptors_list
    save_dict['pickle_features'] = pickle_features_list
    save_dict['skip']=skip
    dump_yaml(save_dict, "config.yaml")
    return 

def get_features_pickle(config):
    """Load features from feature directory."""
    traj_to_workdir = []
    features = []
    weights_metad = []
    times = []
    splits_list = []
    skip = config['skip']
    pickle_features_list = config['pickle_features']
    for c, pickle_features in enumerate(pickle_features_list):
        data = pd.read_pickle(pickle_features)
        if c==0:
            for counter, traj in enumerate(data):
                splits_list.append([])
                # if no split add it once
                frames = traj[::skip].shape[0]
                traj_to_workdir.append(counter+1)

                features += [traj[::skip]]
                weights_metad += [np.ones((frames, 1))]
                times += [np.arange(frames)[:,None]]
        else:
            for counter, traj in enumerate(data):
                features[counter] = np.concatenate([features[counter], traj[::skip]], axis=-1)
    return features, weights_metad, times, splits_list, traj_to_workdir