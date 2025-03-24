# utility functions for rsmlcv

from pytorch_lightning.callbacks.early_stopping import Callback
import numpy as np
import mdtraj as md


import os
import torch
import glob
from pathlib import Path
from ruamel.yaml import YAML
import json
from addict import Dict

yaml = YAML(typ="safe")
yaml.default_flow_style = False

class Encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return json.JSONEncoder.default(self, o)

def decode_object(o):
    if isinstance(o, dict):
        result = {}
        for key, item in o.items():
            result[key] = decode_object(item)
        return result
    elif isinstance(o, list):
        return [decode_object(item) for item in o]
    return o

def load_json(f, is_addict=False):
    with open(f) as handle:
        result = decode_object(json.load(handle))
    if is_addict:
        result = Dict(result)
    return result

def dump_json(o, f):
    check_file_dir(f)
    with open(f, "w") as handle:
        json.dumps(o, handle, cls=Encoder, default=str)

def load_yaml(f, is_addict=False):
    with open(f) as handle:
        result = decode_object(yaml.load(handle))
    if is_addict:
        result = Dict(result)
    return result

def load_yaml_dict(f):
    return load_yaml(f, is_addict=True)

def dump_yaml(o, f, mode="w"):
    check_file_dir(f)
    o = json.loads(json.dumps(o, cls=Encoder))
    with open(f, mode) as handle:
        yaml.dump(o, handle)

def check_file_dir(f):
    Path(f).parent.mkdir(exist_ok=True, parents=True)

def clean_folder(folder_name):
    files = glob.glob(folder_name + "/*")
    for file in files:
        os.remove(file)
    os.rmdir(folder_name)

#### metaD related helper functions ####
def probs_to_fes(probs, units="kcal/mol"):
    """
    Convert probabilities to free energy surface. The reference point is the minimum probability.
    Parameters
    ----------
    probs : numpy.array
        Probabilities.
    units : str, optional
        Units of free energy surface, by default 'kcal/mol'.
    Returns
    -------
    numpy.array
        Free energy surface.
    """
    p_0 = probs[np.where(probs > 0)].min()
    probs[np.where(probs < p_0)] = p_0
    T = 300
    if units == "kcal/mol":
        kT = 0.0019872041  # kcal/mol
        beta = 1 / (kT * T)
        fes = -np.log(probs / p_0) / beta
    elif units == "kJ/mol":
        kT = 0.0083144621  # kJ/mol
        beta = 1 / (kT * T)
        fes = -np.log(probs / p_0) / beta
    return fes


def fes_to_probs(fes, units="kcal/mol"):
    """
    Convert free energy surface to probabilities. The reference point is the minimum free energy.
    Parameters
    ----------
    fes : numpy.array
        Free energy surface.
    units : str, optional
        Units of free energy surface, by default 'kcal/mol'.
    Returns
    -------
    numpy.array
        Probabilities.
    """
    fes = fes - fes.min()
    T = 300
    if units == "kcal/mol":
        kT = 0.0019872041  # kcal/mol
        beta = 1 / (kT * T)
        probs = np.exp(-beta * fes)
    elif units == "kJ/mol":
        kT = 0.0083144621  # kJ/mol
        beta = 1 / (kT * T)
        probs = np.exp(-beta * fes)
    probs = probs / probs.sum()
    return probs


def fes_to_bias(fes, bias_factor):
    """
    Convert free energy surface to bias.
    Parameters
    ----------
    fes : numpy.array
        Free energy surface.
    bias_factor : float
        Bias factor.
    Returns
    -------
    numpy.array
        Bias.
    """
    bias = -(1 - 1 / bias_factor) * fes
    return bias


def bias_to_fes(bias, bias_factor):
    """
    Convert bias to free energy surface.
    Parameters
    ----------
    bias : numpy.array
        Bias.
    bias_factor : float
        Bias factor.
    Returns
    -------
    numpy.array
        Free energy surface.
    """
    fes = -bias / (1 - 1 / bias_factor)
    return fes


from functools import reduce


def add_gaussian(bias, x0, y0, bias_factor, height, noise=False):
    """
    Add a gaussian to the bias according to the rules of metadynamics.
    Parameters
    ----------
    bias : numpy.array
        Current bias.
    x0 : int
        x bin of the center of the gaussian.
    y0 : int
        y bin of the center of the gaussian.
    bias_factor : float
        Bias factor.
    height : float
        Height of the gaussian.
    noise : bool, optional
        Add noise to the center of the gaussian, by default False.
    Returns
    -------
    numpy.array
        New bias.
    """
    if y0 is None:
        flag1d = True
    else:
        flag1d = False
    T = 300
    kT = 0.0083144621  # kJ/mol
    beta = 1 / (kT * T)
    if flag1d:
        current_height = bias[x0]
        x_bins = bias.shape[0]
    else:
        current_height = bias[x0, y0]
        x_bins, y_bins = bias.shape
    factor = height * np.exp(-1 / (bias_factor - 1) * beta * current_height)
    if flag1d:
        x = x0 / x_bins  # which bin is the center of the gaussian
        if noise:
            x += np.random.rand(1)[0] / x_bins - 1 / x_bins / 2
        dist = np.abs(
            np.linspace(0, 1.0, num=x_bins) - x
        )  # distance from center in bins
        scaled_variance = (5 / x_bins) ** 2  # the five is somehow openmm specific
        gaussian = np.exp(-0.5 * dist * dist / scaled_variance)
    else:
        axisGaussians = []
        for (
            x,
            bins,
        ) in zip([x0, y0], [x_bins, y_bins]):
            x = x / bins  # which bin is the center of the gaussian
            if noise:
                x += np.random.rand(1)[0] / bins - 1 / bins / 2
            dist = np.abs(
                np.linspace(0, 1.0, num=bins) - x
            )  # distance from center in bins
            scaled_variance = (5 / bins) ** 2  # the five is somehow openmm specific
            axisGaussians.append(np.exp(-0.5 * dist * dist / scaled_variance))
        # Compute their outer product.
        gaussian = reduce(np.multiply.outer, reversed(axisGaussians)).T
    bias += factor * gaussian
    return bias


def fit_bias_gaussian(
    bias_target,
    initial_height,
    bias_factor,
    max_steps=100000,
    lower=True,
    verbose=False,
    noise=True,
):
    """
    Fit a gaussian to a target bias surface in order to smoothen it, which can be then used as a starting bias for the
    next iteration of metadynamics.
    Parameters
    ----------
    bias_target : numpy.array
        Target bias surface.
    initial_height : float
        Initial height of the gaussian.
    bias_factor : float
        Bias factor.
    max_steps : int, optional
        Maximum number of iterations, by default 100000.
    lower : bool, optional
        Fit only to the lower part of the bias surface, by default True.
    verbose : bool, optional
        Plot the fitting process, by default False.
    noise : bool, optional
        Add noise to the center of the gaussian, by default True.
    Returns
    -------
    numpy.array
        Fitted bias surface.
    """
    if verbose:
        import matplotlib.pyplot as plt
    bias_fit = np.zeros_like(bias_target)
    # make a loop until the difference is as small as possible
    flags = np.ones_like(bias_target, dtype=bool)
    if lower:
        flags[bias_target < initial_height] = False
    counter = 0
    while flags.any() and counter < max_steps:
        # find greatest difference between bias_target and bias_fit
        diff = bias_target - bias_fit
        flags[diff < 0] = False
        diff[~flags] = 0.0
        max_diff = diff.max()
        max_diff_idx = np.where(diff == max_diff)
        # fit a gaussian to the difference
        if bias_fit.ndim == 1:
            x0 = max_diff_idx[0][0]
            y0 = None
        else:
            x0 = max_diff_idx[0][0]
            y0 = max_diff_idx[1][0]
        copy_bias_fit = bias_fit.copy()
        err_pre = np.linalg.norm(bias_target - bias_fit)
        bias_fit = add_gaussian(bias_fit, x0, y0, bias_factor, initial_height, noise)
        err_aft = np.linalg.norm(bias_target - bias_fit)
        if err_aft > err_pre:
            if verbose:
                print("Reached target at ", x0, y0, max_diff)
            if lower:
                bias_fit = copy_bias_fit
            flags[x0, y0] = False
        counter += 1
        if verbose:
            if counter % 1000 == 0:
                plt.imshow(bias_fit)
                plt.show()
                print((bias_target - bias_fit).max())
                plt.imshow(flags)
                plt.show()
    if counter == max_steps:
        print("Max steps reached before convergence, consider increasing max_steps")
    return bias_fit


#### Training stuff ####
class EarlyStopping_CV(Callback):
    """
    Early stopping for reducing the number of input features of the participating CV models.
    Parameters
    ----------
    target_n_feat : list, optional
        Target number of features, by default [30,30]
    verbose : bool, optional
        Verbose, by default True
    save_best_model : bool, optional
        Save the best model, by default False
    tol_save : float, optional
        Tolerance for saving the best model, by default 0.1
    patience : int, optional
        Patience for early stopping, by default 2
    waiting_rounds : int, optional
        Waiting rounds before increasing the penalty for the mask values, by default 2
    """

    def __init__(
        self,
        target_n_feat=[30, 30],
        step_lam=1,
        verbose=True,
    ):
        self.state = {"n_inactive": 0, "batches": 0}
        self.target_n_feat = np.array(target_n_feat)
        self.verbose = verbose
        self.step_lam = step_lam

    def setup(self, trainer, pl_module, stage):
        self.count_threshold = 10
        self.max_increase = 1000
        self.counter = 1
        self.weight_decay = (
            0.02 * pl_module.net.output_dim * np.ones_like(self.target_n_feat)
        )
        for n in range(pl_module.n_cvs):
            pl_module.factors[n] = 0.0
            # trainer.optimizers[0].param_groups[n]['weight_decay'] = weight_decay

    def on_train_batch_end(self, trainer, pt_module, outputs, batch, batch_idx):
        n_inactive = 0
        for n in range(pt_module.n_cvs):
            n_active = pt_module.net.get_active(n)
            trainer.optimizers[0].param_groups[n][
                "weight_decay"
            ] = self.weight_decay[n]
            if n_active <= self.target_n_feat[n]:
                if self.verbose:
                    print("Dim {} reached target perc".format(n))
                pt_module.factors[n] = 0.0
                pt_module.lam_errs[n] = 0.0
                pt_module.subfactors[n] = 0.0
                n_inactive += 1
                trainer.optimizers[0].param_groups[n]["weight_decay"] = 0
            elif (self.counter % self.count_threshold) == 0:
                if self.counter // self.count_threshold <= self.max_increase:
                    # self.weight_decay[n] = 1.02 * self.weight_decay[n]
                    self.weight_decay[n] += self.step_lam
                    trainer.optimizers[0].param_groups[n][
                        "weight_decay"
                    ] = self.weight_decay[n]

        self.counter += 1
        if n_inactive == pt_module.n_cvs:
            if self.verbose:
                print("Reached target features for all dims. Stop training!")
            trainer.should_stop = True
        else:
            trainer.should_stop = False

class EarlyStopping_CV2(Callback):
    """
    Early stopping for reducing the number of input features of the participating CV models.
    Parameters
    ----------
    target_n_feat : list, optional
        Target number of features, by default [30,30]
    verbose : bool, optional
        Verbose, by default True
    save_best_model : bool, optional
        Save the best model, by default False
    tol_save : float, optional
        Tolerance for saving the best model, by default 0.1
    patience : int, optional
        Patience for early stopping, by default 2
    waiting_rounds : int, optional
        Waiting rounds before increasing the penalty for the mask values, by default 2
    """

    def __init__(
        self,
        target_n_feat=[30, 30],
        step_lam=0.1,
        verbose=True,
    ):
        self.state = {"n_inactive": 0, "batches": 0}
        self.target_n_feat = np.array(target_n_feat)
        self.verbose = verbose
        self.step_lam = step_lam

    def setup(self, trainer, pl_module, stage):
        self.count_threshold = 100
        self.max_increase = 10
        self.boundaries = np.array(
            [0.25, 0.1, 0.075, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
        )
        self.counter = 1
        for n in range(pl_module.n_cvs):
            pl_module.factors[n] = 5.0
            pl_module.lam_errs[n] = self.step_lam
            # trainer.optimizers[0].param_groups[n]['weight_decay'] = weight_decay

    def on_train_batch_end(self, trainer, pt_module, outputs, batch, batch_idx):
        n_inactive = 0
        for n in range(pt_module.n_cvs):
            n_active = pt_module.net.get_active(n)   
            if batch_idx % 1 == 0:
                pt_module.log(
                    f"active{n+1}", float(n_active)
                )
            if n_active <= self.target_n_feat[n]:
                if self.verbose:
                    print("Dim {} reached target perc".format(n))
                n_inactive += 1
                pt_module.factors[n] = 0.0
            else:
                # pt_module.factors[n] = 5./((detach_numpy(pt_module.net.weights_norm(n))<self.boundaries).sum()+1)
                pt_module.factors[n] = 5.0 / (
                    np.abs(np.log2(detach_numpy(pt_module.net.weights_norm(n)))) + 1
                )

        self.counter += 1
        if n_inactive == pt_module.n_cvs:
            if self.verbose:
                print("Reached target features for all dims. Stop training!")
            trainer.should_stop = True
        else:
            trainer.should_stop = False

class Save_best_model(Callback):
    """
    Early stopping for reducing the number of input features of the participating CV models.
    Parameters
    ----------
    target_n_feat : list, optional
        Target number of features, by default [30,30]
    verbose : bool, optional
        Verbose, by default True
    save_best_model : bool, optional
        Save the best model, by default False
    tol_save : float, optional
        Tolerance for saving the best model, by default 0.1
    patience : int, optional
        Patience for early stopping, by default 2
    waiting_rounds : int, optional
        Waiting rounds before increasing the penalty for the mask values, by default 2
    """

    def __init__(
        self,
        path_to_save="best_val_model.pt",
        verbose=True,
    ):
        self.state = {"n_inactive": 0, "batches": 0}
        self.path_to_save = path_to_save
        self.verbose = verbose

    def setup(self, trainer, pl_module, stage):
        try:
            self.best_val_value = trainer.callback_metrics[
                "val_score"
            ]  # not existing in first epoch if trainer was destroyed
        except:
            self.best_val_value = 1e10

    def on_validation_epoch_end(self, trainer, pt_module):
        cur_val_save = np.mean(pt_module.validation_step_outputs_enc)
        if cur_val_save <= self.best_val_value:
            pt_module.save(self.path_to_save)
            self.best_val_value = cur_val_save

class WarmUpLR(Callback):
    """
    Early stopping for reducing the number of input features of the participating CV models.
    Parameters
    ----------
    target_n_feat : list, optional
        Target number of features, by default [30,30]
    verbose : bool, optional
        Verbose, by default True
    save_best_model : bool, optional
        Save the best model, by default False
    tol_save : float, optional
        Tolerance for saving the best model, by default 0.1
    patience : int, optional
        Patience for early stopping, by default 2
    waiting_rounds : int, optional
        Waiting rounds before increasing the penalty for the mask values, by default 2
    """

    def __init__(
        self,
        target_lr=0.001,
        start_lr=0.0001,
        steps=100,
        verbose=True,
        counter_start=0,
    ):
        self.state = {"n_inactive": 0, "batches": 0}
        self.target_lr = target_lr
        self.start_lr = start_lr
        self.max_steps = steps
        self.step_counter = counter_start
        self.verbose = verbose

    def on_train_batch_end(self, trainer, pt_module, outputs, batch, batch_idx):
        if self.step_counter <= self.max_steps:
            self.step_counter += 1
            new_lr = self.start_lr + self.step_counter / self.max_steps * (
                self.target_lr - self.start_lr
            )
            for i, param_group in enumerate(trainer.optimizers[0].param_groups):
                if param_group["lr"] < new_lr:
                    param_group["lr"] = new_lr
        else:
            for param_group in trainer.optimizers[0].param_groups:
                param_group["lr"] = self.target_lr
        if self.verbose:
            for param_group in trainer.optimizers[0].param_groups:
                print("Current learning rate is {}".format(param_group["lr"]))


def map_data(x, device="cpu"):
    """Maps the given data x to the specified device. The data can be either a numpy array
    or a torch.Tensor.

    Parameters
    ----------
    x : list of ndarray or torch.Tensor or single ndarray/torch.Tensor
        The data or list of data which should be mapped to the given device
    device : str, optional
        Either GPU name or cpu, by default 'cpu'

    Returns
    -------
    list of torch.Tensor or single torch.Tensor
        The given data on the specified device.
    """

    if isinstance(x, list):
        out = []
        for x_i in x:
            if type(x_i) == np.ndarray:  # if still numpy array
                if x_i.dtype == float:
                    out.append(torch.Tensor(x_i).to(device))
                elif x_i.dtype == int:
                    out.append(torch.LongTensor(x_i).to(device))
                else:
                    "Data type not supported"
            else:
                out.append(x_i.to(device))
    else:
        if isinstance(x, np.ndarray):
            out = torch.Tensor(x).to(device)
        else:
            out = x.to(device)

    return out


def detach_numpy(x):
    return x.detach().cpu().numpy()


def get_frame_traj_number(idx, total_length):
    """Get the trajectory number and the frame number from an continues index."""
    n_trajs = len(total_length)
    traj_n = n_trajs - (idx < total_length).sum()
    # find idx within the whole trajectory
    if traj_n > 0:
        idx = idx - total_length[traj_n - 1]
    return traj_n, idx


############# Auto encoder loss ####################
def loss_auto(x, y, w=None, norm="L2", weights_feat=None):
    """Loss function for auto encoder. Can be used for CVs or MSMs. Either weighted L1 or L2 norm."""
    diff = x - y
    if w is None:
        w = 1 / x.shape[0]
    if weights_feat is None:
        weights_feat = 1
    else:
        weights_feat = weights_feat[None]
    if norm == "L2":
        loss = torch.sum(w * torch.sum(diff * diff * weights_feat, dim=-1, keepdim=True))
    elif norm == "L1":
        loss = torch.sum(w * torch.sum(weights_feat * torch.abs(diff), dim=-1, keepdim=True))
    else:
        loss = torch.sum(w * torch.sum(weights_feat * torch.log(torch.cosh(diff)), dim=-1, keepdim=True))
    return loss


def estimate_fes_cv(
    traj_dir_pattern,
    work_dir,
    stride=1,
    bins=100,
    CV_idx=None,
    bias_factor=None,
    initial_height=None,
    max_steps=1000000,
    noise=True,
    output_dir=None,
):
    """Estimate the free energy surface from a dataset and create matrix.yaml files to display in rshow.
    Parameters
    ----------
    dataset : torch.Dataset
        Dataset containing the trajectories to estimate the free energy surface.
    work_dir : str
        Working directory where the model files are stored and the matrix.yaml files will be created.
    stride : int, optional
        Stride to use when reading the trajectories. The default is 1.
    bins : int, optional
        Number of bins to use when estimating the free energy surface. The default is 100.
    CV_idx : list, optional
        List of indices of the CVs to use. The default is None.
    bias_factor : float, optional
        Bias factor to use in the metadynamics simulation. The default is None. If not None, the FES is fitted with Gaussians, which can be used as a starting bias.
    initial_height : float, optional
        Initial height to use in the metadynamics simulation. The default is None.
    max_steps : int, optional
        Maximum number of steps for fitting the projected FES with Gaussians. The default is 1000000.
    noise : bool, optional
        If True, it will add noise to the projected FES. The default is True.
    Returns
    -------
    F : np.ndarray
        Free energy surface.
    F_smooth : np.ndarray
        Smoothed free energy surface. Only if bias_factor is not None.
    bin_x : np.ndarray
        Bins in the x direction.
    bin_y : np.ndarray
        Bins in the y direction.
    """
    import h5py as h5
    from pathlib import Path
    from .estimator import CVEstimator

    if output_dir is None:
        output_dir = work_dir
    T = 300
    kboltz = 0.008314  # Boltzmann constant in  kJ/(mol*K)
    beta = 1 / (kboltz * T)
    weights=[]
    new_cv_vals = []
    traj_dirs = sorted(glob.glob(traj_dir_pattern))
    for traj_path in traj_dirs:
        with h5.File(traj_path + "path_weights.h5") as F:
            weights_i = np.exp(
                beta * (F["bias"][:] - F["c_t"][:])
            )  # TODO weights already exist in dataset!
        weights.append(weights_i)
        if Path(traj_path + "trajectory_reweight.h5").is_file():
            new_cv_vals_i, _, _ = CVEstimator.estimate_from_file(
                traj_path + "trajectory_reweight.h5",
                work_dir,
                stride=stride,
                CV_idx=CV_idx,
            )
            new_cv_vals.append(new_cv_vals_i)
    weights = np.concatenate(weights, axis=0)
    # weights /= weights.sum() done in proj_cv
    new_cv_vals = np.concatenate(new_cv_vals, axis=0)
    F, bin_x, bin_y = CVEstimator.proj_cv(weights, new_cv_vals, bins)
    F_all = [F]
    if bias_factor is not None:
        from .util import fes_to_bias, fit_bias_gaussian, bias_to_fes

        assert (
            initial_height is not None
        ), "If bias_factor is not None, initial_height must be given."
        bias_reconstruction = fes_to_bias(F, bias_factor)
        bias_new = fit_bias_gaussian(
            bias_reconstruction,
            initial_height=initial_height,
            bias_factor=bias_factor,
            lower=True,
            max_steps=max_steps,
            noise=noise,
        )
        np.save(work_dir + "bias_1_1.npy", bias_new.T)
        F_all.append(bias_to_fes(bias_new, bias_factor))
        # check that the metad file is correct!
        # self.check_metad_file(work_dir, int(bins//5), initial_height, bias_factor)

    # make a plot for rshow
    print("Prepare yaml file to plot the new FES with rshow.")
    dict_matrix = {}
    dict_matrix["trajectories"] = []
    cv_all = []
    n_trajs = len(traj_dirs)
    length = np.zeros(n_trajs)
    for i, traj_path in enumerate(traj_dirs):
        dict_matrix["trajectories"].append(traj_path + "trajectory.h5")
        cv_all_i, _, _ = CVEstimator.estimate_from_file(
            traj_path + "trajectory.h5", work_dir, stride=stride, CV_idx=CV_idx
        )
        cv_all.append(cv_all_i)
        length[i] = np.shape(cv_all_i)[0]
    length = np.cumsum(length)
    cv_all = np.concatenate(cv_all, axis=0)

    for F_ind, F_i in enumerate(F_all):
        dict_matrix["matrix"] = []
        n_x, n_y = F_i.shape
        FES_min = F_i[F_i != -np.inf].min()
        FES_max = F_i[F_i != np.inf].max()
        for x_i in range(n_x):
            list_x = []
            for y_i in range(n_y):
                stat_xy = {}
                # get all frames within the bin
                inside_x = np.logical_and(
                    bin_x[x_i] < cv_all[:, 0], cv_all[:, 0] < bin_x[x_i + 1]
                )
                inside_y = np.logical_and(
                    bin_y[y_i] < cv_all[:, 1], cv_all[:, 1] < bin_y[y_i + 1]
                )
                frames_inside = np.logical_and(inside_x, inside_y)
                frame_numbers = np.where(frames_inside == True)[0]
                if len(frame_numbers) > 0:
                    # get traj and frame number
                    traj_number, frame_number = get_frame_traj_number(
                        frame_numbers[0], length
                    )
                    stat_xy["iFrameTraj"] = [int(frame_number), int(traj_number)]
                    total_list = []
                    for frame_number in frame_numbers:
                        traj_number, frame_number = get_frame_traj_number(
                            frame_numbers[0], length
                        )
                        total_list.append([int(frame_number), int(traj_number)])
                    stat_xy["iFrameTrajs"] = total_list
                eigvalue = F_i[x_i, y_i]
                if eigvalue != -np.inf and eigvalue != np.inf:
                    stat_xy["label"] = "x: {:.3}, y: {:.3}, FES: value: {:.3}".format(
                        (bin_x[x_i + 1] + bin_x[x_i]) * 0.5,
                        (bin_y[y_i + 1] + bin_y[y_i]) * 0.5,
                        eigvalue,
                    )
                    stat_xy["p"] = (eigvalue - FES_min) / (FES_max - FES_min)
                list_x.append(stat_xy)
            dict_matrix["matrix"].append(list_x)
        dump_yaml(dict_matrix, output_dir + "matrix_fes{}.yaml".format(F_ind))
    if bias_factor is not None:
        return F_all[0], F_all[1], bin_x, bin_y
    return F, bin_x, bin_y


def get_bfactor_structure(
    traj_path, cv_file_path, output_path, mode="residue", frame=None
):
    """Given the CV_file.yaml file it fills the bfactors of the structure with the importance weights
    given by the NN. Depending on the mode, it can be residue or atom.
    ----------
    traj_path : str
    Path to the trajectory file
    cv_file_path : str
    Path to the CV_file.yaml file
    output_path : str
    Path where the output structure should be saved
    mode : str
    Mode of the bfactor calculation. Can be residue or atom

    """
    # load the trajectory
    if frame is None:
        traj = md.load(traj_path)
    else:
        traj = md.load_frame(traj_path, index=frame)
    # load the CV file
    cv_file = load_yaml(cv_file_path)
    # get the weights
    weights_feat = cv_file["weights_feat"]
    bfactors = np.zeros((traj.n_frames, traj.n_atoms))
    # get for each residue all atoms
    for feat in weights_feat:
        weight = feat["parameter"]
        atoms_mask = np.array(feat["atom_mask"]).reshape(-1)
        for atom in atoms_mask:
            if mode == "residue":
                res_i = traj.top.residue(traj.top.atom(atom).residue.index)
                for atom_res in res_i.atoms:
                    bfactors[:, atom_res.index] += weight
            else:
                bfactors[:, atom] += weight

    # save the structure
    traj.save_pdb(output_path, bfactors=bfactors)


##### Post analysis of FESs #####
def get_minima_1D(fes, s=1, smooth=True, plot=False):
    """Get minima of a 1D free energy surface. s is the size of the window around each point.
    Returns an array of minima."""
    from itertools import product

    if smooth:
        fes = conv_1d(fes, s)
    enlarged = np.ones((fes.shape[0] + 2 * s))
    enlarged[s:-s] = fes
    segments = []
    fes_values = []
    for index in range(s, fes.shape[0] + s):
        points = np.arange(index - s, index + s + 1)
        result = enlarged[points]
        lowest = np.argmin(result)
        if lowest == s:
            minima = points[lowest]
            segments.append(minima - s)
            fes_values.append(fes[index - s])
    segments = np.array(segments)
    if plot:
        import matplotlib.pyplot as plt

        plt.plot(fes, ".")
        plt.scatter(segments, fes_values, c="r")
        plt.show()
    fes_values = np.array(fes_values)
    sort_id = np.argsort(fes_values)

    return segments[sort_id], fes_values[sort_id]


def conv_1d(fes, s=1):
    enlarged = np.ones((fes.shape[0] + 2 * s))
    enlarged[s:-s] = fes
    enlarged[:s] = fes[0]
    enlarged[-s:] = fes[-1]
    fes_smooth = np.zeros_like(fes)
    for index in range(s, fes.shape[0] + s):
        points = np.arange(index - s, index + s + 1)
        result = enlarged[points]
        fes_smooth[index - s] = np.mean(result)
    return fes_smooth


def get_minima(fes, s=1, plot=False, sigma_smoothing=0):
    """Get minima of a 2D free energy surface. s is the size of the window around each point.
    Returns an array of minima."""
    from itertools import product

    if sigma_smoothing:
        from scipy.ndimage import gaussian_filter

        fes = gaussian_filter(fes, sigma=sigma_smoothing)
    enlarged = np.ones((fes.shape[0] + 2 * s, fes.shape[1] + 2 * s))
    enlarged[s:-s, s:-s] = fes
    segments = []
    fes_values = []
    for i in range(s, fes.shape[0] + s):
        for j in range(s, fes.shape[1] + s):
            index = [i, j]
            points = np.array(
                list(
                    product(
                        np.arange(index[0] - s, index[0] + s + 1),
                        np.arange(index[1] - s, index[1] + s + 1),
                    )
                )
            )
            result = enlarged[points[:, 0], points[:, 1]]
            lowest = np.argmin(result)
            if lowest == 2 * (s**2 + s):
                minima = points[lowest]
                segments.append([minima[0] - s, minima[1] - s])
                fes_values.append(fes[i - s, j - s])
    segments = np.array(segments)
    if plot:
        import matplotlib.pyplot as plt

        plt.clf()
        plt.imshow(fes.T, origin="lower", interpolation="none")
        plt.scatter(segments[:, 0], segments[:, 1], c="r")
        plt.savefig("minima.pdf", bbox_inches="tight")
        plt.show()
    fes_values = np.array(fes_values)
    sort_id = np.argsort(fes_values)

    return segments[sort_id], fes_values[sort_id]


def region_growing(surface, seed, visited, tolerance=0):
    from collections import deque

    rows, cols = surface.shape
    region = [seed]
    queue = deque([seed])
    visited.add(seed)

    while queue:
        i, j = queue.popleft()

        for x, y in [
            (i - 1, j),
            (i, j - 1),
            (i, j + 1),
            (i + 1, j),
            (i + 1, j - 1),
            (i + 1, j + 1),
            (i - 1, j - 1),
            (i - 1, j + 1),
            (i - 2, j),
            (i, j - 2),
            (i, j + 2),
            (i + 2, j),
        ]:
            if 0 <= x < rows and 0 <= y < cols and (x, y) not in visited:
                visited.add((x, y))
                if surface[x, y] > surface[i, j] and surface[x, y] < tolerance:
                    region.append((x, y))
                    queue.append((x, y))
    return region


def find_local_minima_regions(fes, s=1, plot=False, sigma_smoothing=0, tolerance=0):
    if sigma_smoothing:
        from scipy.ndimage import gaussian_filter

        fes = gaussian_filter(fes, sigma=sigma_smoothing)
    local_minima, _ = get_minima(fes, s=s, plot=plot, sigma_smoothing=0)[0]
    visited = set()
    regions = []

    for minimum in local_minima[::-1]:
        minimum = (minimum[0], minimum[1])
        if minimum not in visited:
            region = region_growing(fes, minimum, visited, tolerance=tolerance)
            regions.append(region)
    if plot:
        import matplotlib.pyplot as plt

        plt.imshow(fes.T, origin="lower", interpolation="none")
        for i, region in enumerate(regions):
            # color =np.random.randint(0,255)
            plt.scatter(
                [x[0] for x in region], [x[1] for x in region], s=0.5, c="C{}".format(i)
            )
        plt.scatter(local_minima[:, 0], local_minima[:, 1], c="r")
        plt.show()
    return regions, local_minima


def get_closest_frames(cv_vals, minima, n_samples):
    """Get the closest n_samples frames to each minima."""
    closest_all = []
    for i in range(len(minima)):
        minimum = minima[i]
        distances = np.linalg.norm(cv_vals - minimum, axis=1)
        closest = np.argsort(distances, axis=0)[:n_samples]
        closest_all.append(closest)
    return np.array(closest_all)

################## Plotting functions ###################
# star projections
def project_star(data, func=lambda x: x):
    if type(data) != np.ndarray:
        data = data.numpy()
    dim = data.shape[1]
    # find points on circle
    degree = 2 * np.pi / dim
    state_points = np.zeros((dim, 2))
    for state in range(dim):
        state_points[state] = [np.sin(degree * state), np.cos(degree * state)]
    # now project data
    data_proj = np.sum(func(data[:, :, None]) * state_points[None], 1)

    return data_proj

######################## Create plumed file #########################
def get_torsion_input(atom_mask, counter, ref, CV_name='CV_1'):
    for i in range(len(ref)):
        if np.all(ref.iloc[i]['atominds']==atom_mask):
            name = 'TORSION ATOMS=@{}-{} LABEL={}_feat_{}'.format(ref.iloc[i]['featuregroup'], ref.iloc[i]['resseqs'][0], CV_name, counter)
            break
    return name
def get_weighted_feat(feat, counter, CV_name='CV_1'):
    name = f'MATHEVAL ARG={CV_name}_feat_{counter} FUNC=({feat["transform"]}(x)-{feat["mean"]})/{feat["std"]/feat["parameter"]} LABEL={CV_name}_input_{counter} PERIODIC=NO'
    return name
def get_layer(feat_inputs, feat_output, layer, CV_name='CV_1'):
    W = np.array(layer['W'])
    n_nodes = W.shape[1]
    output_names = []
    name=''
    for n in range(n_nodes):
        name += f'COMBINE LABEL={CV_name}_{feat_output}_{n}_temp ARG={",".join(feat_inputs)} COEFFICIENTS={",".join(map(str,W[:,n]))} PERIODIC=NO\n'
        if 'b' in layer.keys():
            name += f'MATHEVAL ARG={CV_name}_{feat_output}_{n}_temp FUNC=tanh(x+{layer["b"][n]}) LABEL={CV_name}_{feat_output}_{n} PERIODIC=NO\n'
            output_names.append(f'{CV_name}_{feat_output}_{n}')
        else:
            output_names.append(f'{CV_name}_{feat_output}_{n}_temp')
    return name, output_names
def get_inputs(f, weights_feat, ref, CV_name='CV_1'):
    input_feats = []
    for counter in range(len(weights_feat)):
        f.write(get_torsion_input(weights_feat[counter]['atom_mask'], counter, ref, CV_name)+'\n')
    for counter in range(len(weights_feat)):
        f.write(get_weighted_feat(weights_feat[counter], counter, CV_name) +'\n')
        input_feats.append('{}_input_{}'.format(CV_name, counter))
    return input_feats

def create_plumed_input_file(CV_file_list, ref_file_list, output_name, sigma=0.2, height=1, biasfactor=20, temp=300, file='HILLS', pace=1000, label='metad'):
    import pandas as pd
    output_args = []
    with open(output_name, 'w') as f:
        for i, input_file in enumerate(CV_file_list):
            dat = load_yaml(input_file)
            residues = pd.read_pickle(ref_file_list[i])
            CV_name=f'CV_{i}'
            input_feats = get_inputs(f, dat['weights_feat'], residues, CV_name)
            for layer_n in range(len(dat['weights_layer'])):
                lines, input_feats = get_layer(input_feats, f'layer{layer_n}', dat['weights_layer'][layer_n], CV_name)
                f.write(lines)
            output_args.append(input_feats[0])
        f.write(f'METAD ARG={",".join(output_args)} SIGMA={sigma} HEIGHT={height} BIASFACTOR={biasfactor} TEMP={temp} FILE={file} PACE={pace}\
                LABEL={label}')
        f.write(f'PRINT ARG={",".join(output_args)},metad.bias STRIDE=1000 FILE=BIAS.0')
