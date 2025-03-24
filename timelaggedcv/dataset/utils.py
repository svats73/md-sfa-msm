import numpy as np
import h5py
from itertools import product, combinations, chain
import mdtraj
from copy import deepcopy as copy
from addict import Dict
import glob
import pandas as pd

from bisect import bisect_right


def closest_idx_torch(array, value):
    """
    Find index of the element of 'array' which is closest to 'value'.
    The array is first converted to a np.array in case of a tensor.
    Note: it does always round to the lowest one.

    Parameters:
        array (tensor/np.array)
        value (float)

    Returns:
        pos (int): index of the closest value in array
    """
    if type(array) is np.ndarray:
        pos = bisect_right(array, value)
    else:
        pos = bisect_right(array.numpy(), value)
    if pos == 0:
        return 0
    elif pos == len(array):
        return -1
    else:
        return pos - 1


def find_pairs(t, lag, offset=0):
    """
    Searches for all the pairs which are distant 'lag' in time, and returns the weights associated to lag=lag as well as the weights for lag=0.

    Parameters:
        x (tensor): array whose columns are the descriptors and rows the time evolution
        time (tensor): array with the simulation time
        lag (float): lag-time

    Returns:
        x_t (tensor): array of descriptors at time t
        x_lag (tensor): array of descriptors at time t+lag
        w_t (tensor): weights at time t
        w_lag (tensor): weights at time t+lag
    """
    if lag == 0:
        lists_i = np.arange(len(t))
        lists_j = np.arange(len(t))
        delta_tau = np.ones((len(t), 1))
        return [lists_i, lists_j], delta_tau
    # define lists
    x_t = []
    x_lag = []
    w_t = []
    w_lag = []
    # find maximum time idx
    idx_end = closest_idx_torch(t, t[-1] - lag)
    start_j = 0
    # loop over time array and find pairs which are far away by lag
    for i in range(idx_end):
        stop_condition = lag + t[i + 1]
        n_j = 0
        for j in range(start_j, len(t)):
            if (t[j] < stop_condition) and (t[j + 1] > t[i] + lag):
                x_t.append(i + offset)
                x_lag.append(j + offset)
                deltaTau = min(t[i + 1] + lag, t[j + 1]) - max(t[i] + lag, t[j])
                w_lag.append(deltaTau)
                if n_j == 0:  # assign j as the starting point for the next loop
                    start_j = j
                n_j += 1
            elif t[j] > stop_condition:
                break
        for k in range(n_j):
            w_t.append((t[i + 1] - t[i]) / float(n_j))

    return [x_t, x_lag], w_lag

def get_time(bias, c_t, mode="c_t", max_value=10000, stride=1):
    T = 300
    kboltz = 0.008314  # Boltzmann constant in  kJ/(mol*K)
    beta = 1 / (kboltz * T)
    zero = [0]
    if np.ndim(bias) == 2:
        zero = [zero]
    if mode == "c_t":
        exponent = beta * (bias - c_t)
    elif mode == "bias":
        exponent = beta * bias
    elif mode == "log_bias":
        exponent = np.log(beta * bias + 1)
    elif mode == "diff":  # empirically, diff is basically as using tau=0
        exponent = beta * (bias - c_t)
        exponent = exponent[:-1] - exponent[1:]
        exponent = np.concatenate([exponent, zero])
    elif mode == "diff_bias":
        exponent = beta * bias
        exponent = exponent[:-1] - exponent[1:]
        exponent = np.concatenate([exponent, zero])
    elif mode == "diff_inv":
        exponent = beta * (bias - c_t)
        exponent = exponent[1:] - exponent[:-1]
        exponent = np.concatenate([zero, exponent])
    elif mode == "diff_inv_bias":
        exponent = beta * bias
        exponent = exponent[1:] - exponent[:-1]
        exponent = np.concatenate([zero, exponent])
    max_exp = exponent.max()
    b = np.maximum(max_exp / np.log(max_value), 1)
    weights_metad = np.exp(exponent / b)
    time = np.cumsum(weights_metad, axis=0)[::stride] / stride
    if np.ndim(time) == 1:
        time = time[:, None]
    return time


def set_feature_mean_std(features, config):
    """Set the mean and std of the features and creates the feature lists for the neural network."""

    all_features = np.concatenate(features, axis=0)
    all_mean = np.mean(all_features, axis=0)
    all_std = np.std(all_features, axis=0)
    all_std[all_std < 1e-1] = 1e-1  # avoid division by zero
    CV_feature_dicts = config["CV_feature_dicts"]
    cv_nn_mean_list = []
    cv_nn_std_list = []
    features_nn_list = []
    input_dims = []
    indexes_input_list = []
    for CV_i in CV_feature_dicts:
        indexes_cv = CV_i["cv_feature_index"]
        indexes_input_list.append(indexes_cv)
        input_dims.append(len(indexes_cv))
        mean_cv = all_mean[indexes_cv]
        std_cv = all_std[indexes_cv]
        all_cv_features = (
            CV_i["dihedrals"]["features"]
        )
        cv_nn_mean_list.append(mean_cv)
        cv_nn_std_list.append(std_cv)
        for mean_i, std_i, feat_i in zip(mean_cv, std_cv, all_cv_features):
            feat_i["mean"] = mean_i.astype("float64")
            feat_i["std"] = std_i.astype("float64")
        features_nn_list.append(all_cv_features)
    output_dict = config["decoder_dict"]
    indexes_output = output_dict["cv_feature_index"]
    output_dim = len(indexes_output)
    cv_nn_mean_list.append(all_mean[indexes_output])
    cv_nn_std_list.append(all_std[indexes_output])

    return (
        features_nn_list,
        indexes_input_list,
        input_dims,
        indexes_output,
        output_dim,
        cv_nn_mean_list,
        cv_nn_std_list,
    )

def all_pairs_for_all_trajs(dataset):
    all_data_pairs = dataset.dataset.data_pairs
    traj_length = dataset.total_length[:-1]
    n_trajs = dataset.dataset.n_trajs
    all_data_pairs_new = []
    all_kinetic_weights = []
    for i in range(n_trajs):
        if len(dataset.dataset.weights_time[i])>0: # check if there are any pairs in that traj!
            if i==0:
                all_data_pairs_new.append(np.array(all_data_pairs[i]).T)
            else:
                all_data_pairs_new.append(np.array(all_data_pairs[i]).T+traj_length[i-1])
            all_kinetic_weights.append(dataset.dataset.weights_time[i])
    all_data_pairs_new = np.concatenate(all_data_pairs_new, axis=0)
    all_kinetic_weights = np.concatenate(all_kinetic_weights, axis=0)
    return all_data_pairs_new, all_kinetic_weights