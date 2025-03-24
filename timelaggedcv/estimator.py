# create classes to be accessed from the model

import pytorch_lightning as pl

from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from . import CV_autoencoder, CV_NN
from timelaggedcv.dataset import CVfind


from . import load_yaml, dump_yaml
import numpy as np
import torch
from .util import (
    map_data,
    detach_numpy,
    loss_auto,
    EarlyStopping_CV2,
    Save_best_model,
    WarmUpLR,
)
from typing import List


class CVEstimator(pl.LightningModule):
    r"""
    A CV model estimator which can be fit to data optimizing a coarse auto encoder model.

    Parameters
    ----------

    net : CV_autoencoder
        The Auto-encoder net which maps the features to the CV space and back onto the feature space.
    accelerator : str, optional
        The accelerator to use, by default 'gpu'
    devices : int, optional
        The number of devices to use, by default 1
    lr : float, optional
        The learning rate, by default 1e-3
    weight_decay : float, optional
        The weight decay, by default 1e-1
    """

    def __init__(
        self,
        net: CV_autoencoder,
        accelerator: str = "gpu",
        devices: int = 1,
        lr: float = 1e-3,
        lr_decoder: float = None,
        weight_decay: float = 1e-1,
        vamp_loss: str = None,
        vamp_loss_weight: float = 0.0,
        precision_mode: str = "16-mixed",
    ):
        super().__init__()
        self.accelerator = accelerator
        self.devices = devices
        self.net = net
        self.lr = lr
        if lr_decoder is None:
            self.lr_decoder = lr
        else:
            self.lr_decoder = lr_decoder
        self.weight_decay = weight_decay
        self.bottleneck_size = net.bottleneck_dim
        self.precision_mode = precision_mode
        self.lam_errs = np.ones(self.net.n_cv_nn) * 0.1
        self.factors = np.ones(self.net.n_cv_nn)
        self.subfactors = np.ones(self.net.n_cv_nn)
        self.target_n = []
        self.n_cvs = self.net.n_cv_nn
        self.optimizer = None
        self.reached_goal = [False for _ in range(self.net.n_cv_nn)]
        self.validation_step_outputs_loss = []
        self.validation_step_outputs_enc = []
        self.validation_step_outputs_err = []
        self.validation_step_outputs_outer = []
        self.validation_step_outputs_vamp = []
        self.vamp_loss = vamp_loss
        self.vamp_loss_weight = vamp_loss_weight
        print("Created estimator successfully!")

    def training_step(self, batch, batch_idx):
        r"""Performs a partial fit on data. This does not perform any batching.
        Parameters
        ----------
        batch : list
            The training data to feed through all the CVs, weights, and target data.
        batch_idx : int
            The index of the batch.

        Returns
        -------
        loss_value : torch.Tensor
        """
        loss_enc, loss_outer, loss_vamp = self.net.estimate_auto_loss(batch)
        loss_value = loss_enc
        loss_enc_item = loss_enc.item()
        for n_cv in range(self.n_cvs):
            err = self.net.weights_norm(n_cv)
            err_item = detach_numpy(err)
            if self.lam_errs[n_cv] > 0:
                # to get it down to specific percent
                fac = self.lam_errs[n_cv] * loss_enc_item / err_item
                if len(self.target_n) > 0:
                    if self.net.get_active(n_cv) <= self.target_n[n_cv]:
                        fac = 0.0  # stop penalizing if reached goal
            else:
                fac = 0.0
            loss_value += fac * self.factors[n_cv] * err
            index_nn_cv = self.net.index_nn_cv[n_cv]
            loss_outer_item = loss_outer[index_nn_cv].item()
            if loss_outer_item > 0:
                # fac_outer = 0.1 * loss_enc_item / loss_outer_item
                fac_outer = self.net.output_dim * 1
                loss_value += fac_outer * loss_outer[index_nn_cv]
        if self.vamp_loss is not None:
            # fac = self.vamp_loss_weight * loss_enc_item / vamp_item
            fac = self.vamp_loss_weight/self.n_cvs*self.net.output_dim
            loss_value += fac * loss_vamp
            self.log(self.vamp_loss, loss_vamp)
        self.log("train_loss", loss_value)
        self.log("train_score", loss_enc_item)            
        return loss_value

    def validation_step(self, batch, batch_idx):
        r"""Performs validation on data. This does not perform any batching.

        Parameters
        ----------
        batch : list
            The validation data to feed through all the CVs, weights, and target data.
        batch_idx : int
            The index of the batch.

        Returns
        -------
        loss_value : np.ndarray
            The total loss value
        loss_enc_item : np.ndarray
            The loss value of the autoencoder
        err_item_list : List[np.ndarray]
            The loss value for the mask weights.

        """
        loss_enc, loss_outer, loss_vamp = self.net.estimate_auto_loss(batch)
        loss_value = loss_enc
        loss_enc_item = loss_enc.item()
        err_item_list = []
        loss_outer_list = []
        for n_cv in range(self.n_cvs):
            err = self.net.weights_norm(n_cv)
            err_item = detach_numpy(err)
            err_item_list.append(err_item)
            if self.lam_errs[n_cv] > 0:
                fac = (
                    self.lam_errs[n_cv] * loss_enc_item / err_item
                )  # to get it down to specific percent
            else:
                fac = 0.0
            if len(self.target_n) > 0:
                if self.net.get_active(n_cv) <= self.target_n[n_cv]:
                    fac = 0.0  # stop penalizing if reached goal
            loss_value += fac * self.factors[n_cv] * err
            index_nn_cv = self.net.index_nn_cv[
                n_cv
            ]  # get the index of that nn_cv not a hand designed one
            loss_outer_item = loss_outer[index_nn_cv].item()
            loss_outer_list.append(loss_outer_item)
            if loss_outer_item > 0:
                # fac_outer = 0.1 * loss_enc_item / loss_outer_item
                fac_outer = self.net.output_dim * 1
                loss_value += fac_outer * loss_outer[index_nn_cv]
        if self.vamp_loss is not None:
            vamp_item = detach_numpy(loss_vamp.type(torch.float32))
            # fac = self.vamp_loss_weight * loss_enc_item / vamp_item
            fac = self.vamp_loss_weight/self.n_cvs*self.net.output_dim
            loss_value += fac * loss_vamp
            self.validation_step_outputs_vamp.append(vamp_item)
        self.validation_step_outputs_loss.append(detach_numpy(loss_value))
        self.validation_step_outputs_err.append(err_item_list)
        self.validation_step_outputs_enc.append(loss_enc_item)
        self.validation_step_outputs_outer.append(loss_outer_list)            
        return detach_numpy(loss_value), loss_enc_item, err_item_list, loss_outer_list

    def on_validation_epoch_end(self):
        r"""Performs mean estimation and logging of validation scores at the end of epoch"""
        loss_mean = np.mean(self.validation_step_outputs_loss)
        loss_enc_item = np.mean(self.validation_step_outputs_enc)
        err_item = np.mean(self.validation_step_outputs_err, axis=0)
        loss_outer = np.mean(self.validation_step_outputs_outer, axis=0)
        for i in range(len(err_item)):
            self.log(f"weight_norm{i+1}", err_item[i])
            self.log(f"outer_loss{i+1}", loss_outer[i])
        self.log("val_loss", loss_mean)
        self.log("val_score", loss_enc_item)
        self.validation_step_outputs_loss.clear()
        self.validation_step_outputs_err.clear()
        self.validation_step_outputs_enc.clear()
        self.validation_step_outputs_outer.clear()
        if self.vamp_loss is not None:
            loss_vamp = np.mean(self.validation_step_outputs_vamp)
            self.log("val_"+self.vamp_loss, loss_vamp)
        return loss_mean, loss_enc_item, err_item, loss_outer

    def configure_optimizers(self):
        r"""Configures the optimizer. This is called by pytorch lightning, so to use one optimizer for the whole pipeline."""
        if self.optimizer is None:
            decay = []
            no_decay = [[] for _ in range(len(self.net.cv_nets))]
            for n_model, model in enumerate(self.net.cv_nets):
                for name, param in model.named_parameters():
                    if "weight" in name or "bias" in name:
                        decay.append(param)
                    else:
                        no_decay[n_model].append(param)
            decoder = []
            for name, param in self.net.decoder.named_parameters():
                if "weight" in name or "bias" in name:
                    decoder.append(param)
            param_list = []
            for param in no_decay:
                param_list.append({"params": param, "weight_decay": 0, "lr": self.lr})
            param_list.append(
                {"params": decay, "weight_decay": self.weight_decay, "lr": self.lr}
            )
            param_list.append({"params": decoder, "weight_decay": self.weight_decay})
            self.optimizer = torch.optim.AdamW(
                param_list,
                lr=self.lr_decoder,
            )
        return self.optimizer

    def configure_optimizers_without_decoder(self, decoder=False):
        r"""Configures the optimizer. This is called by pytorch lightning, so to use one optimizer for the whole pipeline."""
        decay = []
        no_decay = [[] for _ in range(len(self.net.cv_nets))]
        for n_model, model in enumerate(self.net.cv_nets):
            for name, param in model.named_parameters():
                if "weight" in name or "bias" in name:
                    decay.append(param)
                else:
                    no_decay[n_model].append(param)
        param_list = []
        for param in no_decay:
            param_list.append({"params": param, "weight_decay": 0, "lr": self.lr})
        param_list.append({"params": decay, "weight_decay": 0, "lr": self.lr})
        if decoder:
            decoder = []
            for name, param in self.net.decoder.named_parameters():
                if "weight" in name or "bias" in name:
                    decoder.append(param)
            param_list.append({"params": decoder, "weight_decay": self.weight_decay})

        self.optimizer = torch.optim.AdamW(
            param_list,
            lr=self.lr_decoder,
        )

    def save(self, path: str):
        r"""Save the current estimator at path.

        Parameters
        ----------
        path: str
            The path where to save the model.

        """
        save_dict = {
            "net_state_dict": self.net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
            if self.optimizer is not None
            else None,
        }
        torch.save(save_dict, path)

    def load(self, path: str):
        r"""Load the estimator from path.
         The architecture needs to fit!

        Parameters
        ----------
        path: str
             The path where the model is saved.
        """

        checkpoint = torch.load(path, map_location=self.device)
        if self.optimizer is None:
            self.configure_optimizers()
        self.net.load_state_dict(checkpoint["net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def count_reached_target(self, target_n_feat):
        """Helper function to count how many CVs have reached the target number of features."""
        counter_target = 0
        for n in range(self.n_cvs):
            n_active = self.net.get_active(n)
            # kernel_n = np.abs(self.net.get_kernel(n))
            if n_active <= target_n_feat[n]:
                counter_target += 1
        return counter_target

    def fit_routine(
        self,
        target_n_feat: List,
        data_loader: torch.utils.data.DataLoader,
        validation_loader: torch.utils.data.DataLoader,
        patience: int = 2,
        step_lam: float = 4.0,
        load_model: bool = False,
        pretrain_model: bool = True,
        verbose: bool = False,
        max_steps=int(1e6),
        max_rounds=10,
    ):
        r"""Fitting routine:
            1. Pretrain the model with the mask weights fixed.
            2. Fit the model with the mask weights trainable and punish the large weights.
            3. Repeat 2. until the target number of features is reached for all CVs.
            4. Train the model without penalizing the large weights until convergence.

        Parameters
        ----------
        target_n_feat: list
            The target number of features for each CV.
        data_loader: torch.utils.data.DataLoader
            The data loader for the training data.
        validation_loader: torch.utils.data.DataLoader
            The data loader for the validation data.
        patience: int, optional
            The patience for the early stopping. The default is 2.
        waiting_rounds: int, optional
            The number of rounds to wait before increasing the penalty. The default is 2.
        step_lam: float, optional
            The step size for increasing the penalty. The default is 0.1.
        tol_save: float, optional
            The tolerance for saving the model. The default is 0.1.
        save_inter_model: bool, optional
            Whether to save the intermediate models. The default is True.
        load_model: bool, optional
            Whether to load a model. The default is False.
        pretrain_model: bool, optional
            Whether to pretrain the model. The default is True.
        verbose: bool, optional
            Whether to print the progress. The default is False.

        Returns
        -------
        void
        """
        if type(target_n_feat) == int:
            target_n_feat = [
                target_n_feat for _ in range(self.n_cvs)
            ]  # assume all the same
            self.target_n = target_n_feat
        self.factors = np.ones(self.n_cvs)
        self.subfactors = np.ones(self.n_cvs)
        # fit till convergence
        flag = True
        self.lr = 0.0001
        if load_model:
            self.load("./model_full.pt")
            print("Loaded model")
            counter_warmup = 0
        else:
            if pretrain_model:
                self.log("val_loss", 1000)
                self.net.switch_mask_training(False)
                self.lam_errs = np.zeros(self.n_cvs)
                self.trainer = pl.Trainer(
                    accelerator=self.accelerator,
                    devices=self.devices,
                    max_steps=max_steps,
                    precision=self.precision_mode,
                )
                self.trainer.callbacks[0].disable()  # disable progress bar
                self.trainer.callbacks.pop(2)  # remove modelckeckpoint
                self.trainer.callbacks += [
                    EarlyStopping(monitor="val_loss", mode="min", patience=patience),
                    WarmUpLR(0.001, self.lr_decoder, 1000, verbose=verbose),
                ]
                self.trainer.fit(self, data_loader, validation_loader)
                counter_warmup = self.trainer.callbacks[-1].step_counter
        self.save("./model_full.pt")
        self.plot_cv(data_loader, skip=100)
        # the first time set threshold based on the worst active
        # self.set_mask()
        self.lam_errs = np.array([step_lam, step_lam])
        print("Set the threshold and enact constraints")
        self.net.switch_mask_training(True)
        counter_while = 0
        # for g in self.optimizer.param_groups:
        #     g['lr'] = self.lr
        while flag:
            self.trainer = pl.Trainer(
                accelerator=self.accelerator,
                devices=self.devices,
                max_steps=max_steps,
                precision=self.precision_mode,
            )
            self.trainer.callbacks[0].disable()  # disable progress bar
            self.trainer.callbacks.pop(2)  # remove modelckeckpoint
            self.trainer.callbacks += [
                EarlyStopping_CV2(
                    target_n_feat,
                    verbose=verbose,
                    step_lam=step_lam,
                ),
                WarmUpLR(
                    0.001, self.lr_decoder, 1000, verbose=verbose, counter_start=counter_warmup
                ),
            ]
            self.trainer.fit(self, data_loader, validation_loader)
            self.save("./temp_path_model_constraint.pt")
            self.plot_cv(data_loader, skip=100)
            counter_target = self.count_reached_target(target_n_feat)
            if counter_target == self.n_cvs or counter_while > max_rounds:
                flag = False
            else:
                # increase penalty
                print("Increasing the lam penalty for another round")
                self.lam_errs = self.lam_errs * 1.5
                counter_while += 1
        self.lr = 0.001
        if counter_target == self.n_cvs:
            print("Train till convergence without contraints")
            # for g in self.optimizer.param_groups:
            #     g['lr'] = self.lr
            self.lam_errs = np.zeros_like(self.lam_errs)
            self.factors = np.zeros_like(self.factors)
            self.subfactors = np.zeros_like(self.subfactors)
            self.save("./model_final_best.pt")
            self.trainer = pl.Trainer(
                accelerator=self.accelerator,
                devices=self.devices,
                precision=self.precision_mode,
            )
            self.trainer.callbacks[0].disable()  # disable progress bar
            self.trainer.callbacks.pop(2)  # remove modelckeckpoint
            self.trainer.callbacks += [
                EarlyStopping(
                    monitor="val_loss",
                    mode="min",
                    patience=patience,  # * waiting_rounds # do not wait too long.
                ),
                Save_best_model("./model_final_best.pt"),
            ]
            self.trainer.fit(self, data_loader, validation_loader)
            print("Loading best model")
            self.load("./model_final_best.pt")
        else:
            print(
                "Not reached convergence. Could try to train with larger lam to enforce sparsity"
            )
        self.plot_cv(data_loader, skip=100)
        self.save("./model_final.pt")
        self.log_active_features(data_loader.dataset.dataset)
        return

    def fit_finetune_student(
        self, net_student: CV_autoencoder, data_loader, validation_loader, patience=2
    ):
        """Fit the student model to the data, but keep the weights of the teacher fixed."""
        # set the net to student net
        self.net.cv_nets = net_student.cv_nets
        # reset the optimizer
        self.configure_optimizers_without_decoder(decoder=False)
        self.lam_errs = np.zeros_like(self.lam_errs)
        self.factors = np.zeros_like(self.factors)
        self.subfactors = np.zeros_like(self.subfactors)
        self.save("./model_final_best.pt")
        self.trainer = pl.Trainer(
            accelerator=self.accelerator,
            devices=self.devices,
            precision=self.precision_mode,
        )
        self.trainer.callbacks[0].disable()  # disable progress bar
        self.trainer.callbacks.pop(2)  # remove modelckeckpoint
        self.trainer.callbacks += [
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=patience,  # do not wait too long.
            ),
            Save_best_model("./model_final_best.pt"),
        ]
        self.trainer.fit(self, data_loader, validation_loader)
        self.load("./model_final_best.pt")
        self.configure_optimizers_without_decoder(decoder=True)
        self.trainer = pl.Trainer(
            accelerator=self.accelerator,
            devices=self.devices,
            precision=self.precision_mode,
        )
        self.trainer.callbacks[0].disable()  # disable progress bar
        self.trainer.callbacks.pop(2)  # remove modelckeckpoint
        self.trainer.callbacks += [
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=patience,  # do not wait too long.
            ),
            Save_best_model("./model_final_best.pt"),
        ]
        self.trainer.fit(self, data_loader, validation_loader)
        print("Loading best model")
        self.load("./model_final_best.pt")
        self.log_active_features(data_loader.dataset.dataset)

    def plot_cv(self, dataloader, skip=100):
        """Plot the CVs given a dataloader."""
        import matplotlib.pyplot as plt

        bott_all = self.pred_cv(dataloader)
        if bott_all.shape[1] == 2:
            plt.scatter(bott_all[::skip, 0], bott_all[::skip, 1])
        else:
            plt.plot(bott_all[::skip, 0], ".")
        plt.savefig('bottleneck_scatter')
        plt.clf()

    def get_bfactors(self, dataset):
        """Estimates the weights of each atom for each CV. The array can be used to visualize which
        atoms are important for each CV, e.g. by saving them as a bfactor in a pdb file.
        """
        traj_ref = dataset.traj_ref
        bfactors = np.zeros((self.n_cvs, traj_ref.n_atoms))
        for cv_n in range(self.n_cvs):
            kernel = self.net.get_kernel(cv_n)
            for i, feat in enumerate(dataset.feat_nn_list[cv_n]):
                if kernel[i] > 0:
                    atoms_list = feat["atom_mask"]
                    for atom in atoms_list:
                        bfactors[cv_n, atom] += kernel[i]
        return bfactors

    def log_active_features(self, dataset):
        import matplotlib.pyplot as plt

        for n in range(self.net.n_cv_nn):
            if type(self.net.cv_nets[n]) == CV_NN:
                list_cv, weights_cv = self.net.cv_nets[n].get_feature_weights(
                    dataset, n
                )
                plt.clf()
                fig = plt.figure()
                x_feature = [
                    feat["feature"]
                    + " "
                    + feat["transform"]
                    + " "
                    + " ".join([str(res_i) for res_i in feat["residues"]])
                    for feat in list_cv
                ]
                y_parameter = [feat["parameter"] for feat in list_cv]
                plt.bar(range(len(x_feature)), y_parameter)
                plt.xticks(range(len(x_feature)), x_feature, rotation=20)
                plt.ylabel("Weight", fontsize=12)
                plt.savefig(f"Feature_weights_{n}")
                plt.clf()

    def pred_cv(self, dataloader):
        """Predict the CVs given a dataloader."""
        bott_all = []
        self.net.eval()
        with torch.no_grad():
            for batch in dataloader:
                x_t = []
                for batch_i in batch[: self.net.bottleneck_dim]:
                    x_t.append(map_data(batch_i, device=self.device))
                bott_all.append(self.net.bottleneck(x_t).detach().cpu().numpy())
        bott_all = np.concatenate(bott_all, axis=0)
        return bott_all

    def estimate_cv_from_dataset(self, dataset, batchsize=None):
        """Estimate the CVs given a dataset. It creates the dataloader internally."""
        from torch.utils.data import DataLoader

        if batchsize is None:
            batchsize = len(dataset)
            if batchsize > 10000:
                batchsize = 10000
            print("Setting batchsize to max value: {}".format(batchsize))
        dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=False)
        bott_all = self.pred_cv(dataloader)
        return bott_all

    def get_optimal_feats(
        self,
        workdir,
        dataset: CVfind,
        plot_CV_space=True,
        skip=100,
        batchsize=None,
        safety_percent=0.1,
    ):
        """Estimates the files with the optimal features and NN variables for each CV and creates the
        metad.yaml file to start a simulation."""
        from .util import create_plumed_input_file
        bott_all = self.estimate_cv_from_dataset(dataset, batchsize=batchsize)
        if plot_CV_space:
            if self.net.bottleneck_dim > 2:
                print("Only plotting the first two dims!")
            import matplotlib.pyplot as plt

            plt.clf()
            fig = plt.figure()
            if self.net.bottleneck_dim == 2:
                plt.scatter(bott_all[::skip, 0], bott_all[::skip, 1])
                plt.xlabel("NN_CV 1", fontsize=14)
                plt.ylabel("NN_CV 2", fontsize=14)
            else:
                plt.plot(bott_all[::skip, 0], ".")
                plt.ylabel("NN_CV 1", fontsize=14)
            plt.savefig(workdir + "bottleneck_plot_before.pdf", bbox_inches="tight")
            plt.clf()
            # plt.show()
        min_values, max_values = bott_all.min(0), bott_all.max(0)
        diff = max_values - min_values
        rel = diff * safety_percent
        max_values += rel
        min_values -= rel
        self.net.get_opt_feat(workdir, dataset.dataset, max_values, min_values)
        # create a plumed file for the next run
        CV_list = ['CV_file0.yaml', 'CV_file1.yaml']
        ref_file_list = dataset.dataset.config['pickle_descriptors']
        output_name = 'plumed.dat'
        create_plumed_input_file(CV_list, ref_file_list, output_name)

        self.save(workdir + "model.pt")

    @staticmethod
    def estimate_from_file(traj_path, workdir, cv_list=['CV_file0.yaml', 'CV_file1.yaml'], stride=1, CV_idx=None):
        """Estimate the CVs from a trajectory file. It uses the metad.yaml file to know the CVs. No pytorch model is used.
        It only uses the CV files."""
        import mdtraj as md
        import h5py

        traj = md.load_hdf5(traj_path, stride=stride)
        # atoms_backbone = [atom.index for atom in traj.top.atoms if atom.is_backbone]
        # traj = traj.superpose(traj, frame=0, atom_indices=atoms_backbone)
        cv_values = []
        min_vals = []
        max_vals = []
        for cv_i in cv_list:
            data = load_yaml(cv_i)
            component = 0
            acti = data["acti"]
            max_vals.append(data["cv_max"][component])
            min_vals.append(data["cv_min"][component])
            weights_feat = data["weights_feat"]
            weights_layer = data["weights_layer"]  # stored NN weights
            feats = []
            for feat_i in weights_feat:
                param = feat_i["parameter"]
                if feat_i["feature"] == "dihedrals":
                    feat_traj_temp = md.compute_dihedrals(
                        traj, [feat_i["atom_mask"]]
                    )
                    if feat_i["transform"] == "cos":
                        feat = np.cos(feat_traj_temp)
                    else:
                        feat = np.sin(feat_traj_temp)
                elif feat_i["feature"] == "distance":
                    feat_traj_temp = md.compute_distances(
                        traj, [feat_i["atom_mask"]]
                    )
                    if feat_i["transform"] == "inv":
                        feat = 1 / feat_traj_temp
                    elif feat_i["transform"] == "cn":
                        feat = 1 / (1 + (feat_traj_temp/feat_i["decay_distance_nm"])**6)
                    else:
                        feat = feat_traj_temp
                elif feat_i["feature"] == "contacts":
                    com_lig = md.compute_center_of_geometry(
                        traj.atom_slice(feat_i["atom_mask"][0])
                    )
                    com_prot = md.compute_center_of_geometry(
                        traj.atom_slice(feat_i["atom_mask"][1])
                    )
                    distance = np.linalg.norm(
                        com_lig - com_prot, axis=-1, keepdims=True
                    )
                    if feat_i["transform"] == "inv":
                        feat = 1 / distance
                    elif feat_i["transform"] == "cn":
                        feat = 1 / (1 + (distance/feat_i["decay_distance_nm"])**6)
                    else:
                        feat = distance

                else:
                    raise NotImplementedError("Feature not implemented!")
                if "mean" in feat_i.keys():
                    feat = (feat - feat_i["mean"]) / feat_i["std"]
                feats.append(feat * param)

            feats = np.concatenate(feats, axis=-1)
            if acti == "Exp":
                acti = np.exp
            elif acti == "Tanh":
                acti = np.tanh
            elif acti == "Softplus":
                acti = lambda x: np.log(1 + np.exp(x))
            for layer in weights_layer:
                W = np.array(layer["W"])
                feats = feats @ W
                if "b" in layer.keys():
                    b = np.array(layer["b"])
                    feats = acti(feats + b[None, :])  # b has no time axis!
            cv_values.append(feats)
        cv_values = np.concatenate(cv_values, axis=-1)

        return cv_values, min_vals, max_vals

    @staticmethod
    def proj_cv(
        weights,
        cv,
        bins=10,
        units="kJ/mol",
        min_vals=None,
        max_vals=None,
        axis=None,
        frames_included=None,
        safety_percent=0.1,
    ):
        """Project the data to estimated CVs and returns the FES."""
        import matplotlib.pyplot as plt
        from .util import probs_to_fes

        assert (
            weights.shape[0] == cv.shape[0]
        ), "weights and cv do not have matching frames"
        dim = cv.shape[1]
        steps = bins // 5
        if min_vals is None:
            min_vals = cv.min(0).astype("float64")
            max_vals = cv.max(0).astype("float64")
            diff = (
                max_vals - min_vals
            )  # add 10% here as well as when estimating optimal features
            rel = diff * safety_percent
            max_vals += rel
            min_vals -= rel
        # normalize weights
        if frames_included is not None:
            weights = weights[frames_included]
            print("Only using {} frames for projection".format(weights.shape[0]))
            cv = cv[frames_included]
        weights /= weights.sum()
        if dim == 1:
            c, bin_x = np.histogram(
                cv[:, 0],
                bins=bins,
                range=[min_vals[0], max_vals[0]],
                weights=weights[:, 0],
            )
            F = probs_to_fes(c, units=units)
            if axis is None:
                plt.clf()
                fig = plt.figure()
                plt.plot(bin_x[:-1], F)
                # plt.xticks(np.arange(0, bins, steps), np.round(bin_x[:-1:steps], 1))
                plt.xlabel("Projected CV", fontsize=14)
                plt.ylabel("Free energy [kJ/mol]", fontsize=14)
                plt.savefig("ProjectedFES.pdf", bbox_inches="tight")
                plt.clf()
            else:
                axis.plot(bin_x[:-1], F)
                # plt.xticks(np.arange(0, bins, steps), np.round(bin_x[:-1:steps], 1))
                axis.set_xlabel("Projected CV", fontsize=14)
                axis.set_ylabel("Free energy [kJ/mol]", fontsize=14)
            return F, bin_x, None
        elif dim == 2:
            c, bin_x, bin_y = np.histogram2d(
                cv[:, 0],
                cv[:, 1],
                bins=bins,
                range=[[min_vals[0], max_vals[0]], [min_vals[1], max_vals[1]]],
                weights=weights[:, 0],
            )
            # c, bin_x, bin_y = np.histogram2d(cv[:,0], cv[:,1], bins=bins, weights=weights[:,0])
            F = probs_to_fes(c, units=units)
            plt.clf()
            fig = plt.figure()
            pos = plt.imshow(
                F.T,
                origin="lower",
                interpolation="hanning",
                extent=[bin_x[0], bin_x[-1], bin_y[0], bin_y[-1]],
                aspect="auto",
            )
            # plt.xticks(np.arange(0, bins, steps), np.round(bin_x[:-1:steps], 1))
            # plt.yticks(np.arange(0, bins, steps), np.round(bin_y[:-1:steps], 1))
            plt.xlabel("CV 1", fontsize=14)
            plt.ylabel("CV 2", fontsize=14)
            cbar = plt.colorbar(pos)
            cbar.set_label("Free energy [kJ/mol]", rotation=270, labelpad=20)
            cbar.ax.tick_params(labelsize=8)
            plt.savefig("ProjectedFES.pdf", bbox_inches="tight")
            plt.clf()
            return F, bin_x, bin_y

    def check_convergence(
        self,
        workdirs_list1,
        workdirs_list2,
        work_dir,
        bins=10,
        stride=1,
        CV_idx=None,
        CN=False,
    ):
        """Check convergence of FES from two sets of simulations, where the model can be specified.
        Parameters
        ----------
        workdirs_list1 : list of strings
            List of paths to the work directories of the first set of simulations.
        workdirs_list2 : list of strings
            List of paths to the work directories of the second set of simulations.
        work_dir : string
            Path to the work directory of the model.
        bins : int, optional
            Number of bins to use for the histogram. The default is 10.
        stride : int, optional
            Stride to use for the histogram. The default is 1.
        CV_idx : list of ints, optional
            List of indices of the CVs to use. The default is None. It assumes that the CV was stored at that index.
        CN : bool, optional
            If True, it will use the CN CVs. The default is False.
        Returns
        -------
        diff_F : np.ndarray
            Difference in the FES between the two sets of simulations.
        F1 : np.ndarray
            FES of the first set of simulations.
        F2 : np.ndarray
            FES of the second set of simulations."""
        import h5py as h5
        import matplotlib.pyplot as plt
        import mdtraj as md
        T = 300
        kboltz = 0.008314  # Boltzmann constant in  kJ/(mol*K)
        beta = 1 / (kboltz * T)
        weights1 = []
        new_cv_vals1 = []
        weights2 = []
        new_cv_vals2 = []
        for traj_path in np.unique(workdirs_list1 + workdirs_list2):
            with h5.File(traj_path + "path_weights.h5") as F:
                weights_i = np.exp(beta * (F["bias"][:] - F["c_t"][:]))
                if len(weights_i.shape) == 1:
                    weights_i = weights_i[:, None]
            new_cv_vals_i, _, _ = self.estimate_from_file(
                traj_path + "trajectory_reweight.h5",
                work_dir,
                stride=stride,
                CV_idx=CV_idx,
            )
            if traj_path in workdirs_list1:
                weights1.append(weights_i)
                new_cv_vals1.append(new_cv_vals_i)
            if traj_path in workdirs_list2:
                weights2.append(weights_i)
                new_cv_vals2.append(new_cv_vals_i)

        weights1 = np.concatenate(weights1, axis=0)
        new_cv_vals1 = np.concatenate(new_cv_vals1, axis=0)
        weights2 = np.concatenate(weights2, axis=0)
        new_cv_vals2 = np.concatenate(new_cv_vals2, axis=0)

        cv_all = np.concatenate([new_cv_vals2, new_cv_vals1], axis=0)
        max_vals = np.max(cv_all, axis=0)
        if CN:
            new_cv_vals1 /= max_vals
            new_cv_vals2 /= max_vals
        min_vals = np.min(cv_all, axis=0)

        ret1 = self.proj_cv(
            weights1,
            new_cv_vals1,
            bins=bins,
            min_vals=min_vals,
            max_vals=max_vals,
            axis=None,
        )
        ret2 = self.proj_cv(
            weights2,
            new_cv_vals2,
            bins=bins,
            min_vals=min_vals,
            max_vals=max_vals,
            axis=None,
        )
        F1 = ret1[0]
        F2 = ret2[0]

        diff_F = F1 - F2
        if cv_all.shape[1] == 2:
            steps = bins // 5
            plt.clf()
            pos = plt.imshow(diff_F.T, origin="lower", interpolation="hanning")
            plt.xticks(np.arange(0, bins, steps), np.round(ret1[1][:-1:steps], 1))
            plt.yticks(np.arange(0, bins, steps), np.round(ret1[2][:-1:steps], 1))
            plt.xlabel("CV 1", fontsize=14)
            plt.ylabel("CV 2", fontsize=14)
            cbar = plt.colorbar(pos)
            cbar.set_label("Free energy difference [kJ/mol]", rotation=270, labelpad=20)
            cbar.ax.tick_params(labelsize=8)
            plt.savefig("Convergence_FES.pdf", bbox_inches="tight")
            plt.clf()
        else:
            not_ind = diff_F != np.inf
            plt.clf()
            plt.plot(ret1[1][:-1][not_ind], diff_F[not_ind])
            # plt.xticks(np.arange(0, bins, steps), np.round(bin_x[:-1:steps], 1))
            plt.xlabel("Projected CV", fontsize=14)
            plt.ylabel("Free energy [kJ/mol]", fontsize=14)
            plt.savefig("ProjectedFES.pdf", bbox_inches="tight")
            plt.clf()
        return diff_F, F1, F2

    def estimate_fes_cv(
        self,
        dataset: CVfind,
        work_dir,
        bins=100,
        bias_factor=None,
        initial_height=None,
        max_steps=1000000,
        noise=True,
        batchsize=10000,
        safety_percent=0.1,
    ):
        """Estimate the free energy surface from a dataset and create matrix.yaml files to display in rshow.
        Parameters
        ----------
        dataset : torch.Dataset
            Dataset containing the trajectories to estimate the free energy surface.
        work_dir : str
            Working directory where the model files are stored and the matrix.yaml files will be created.
        bins : int, optional
            Number of bins to use when estimating the free energy surface. The default is 100.
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
        if self.net.bottleneck_dim > 2:
            print("Only possible with less than 3 CVs!")
            return
        weights = dataset.dataset.metad_weights
        weights = np.concatenate(weights, axis=0)
        # weights /= weights.sum() done in proj_cv
        new_cv_vals = self.estimate_cv_from_dataset(
            dataset, batchsize=batchsize
        )  # TODO Add mask which runs you want consider for bias
        if dataset.dataset.bias_mask is not None:
            frames_included = []
            start = 0
            for i, traj_dir in enumerate(dataset.dataset.traj_to_workdir):
                end = dataset.total_length[i]
                if traj_dir in dataset.dataset.bias_mask:
                    frames_included.append(np.arange(start, end))
                start = end
            frames_included = np.concatenate(frames_included, axis=0)
        else:
            frames_included = None
        F, bin_x, bin_y = self.proj_cv(
            weights,
            new_cv_vals,
            bins,
            frames_included=frames_included,
            safety_percent=safety_percent,
        )
        if bin_y is None:
            flag1D = True
        else:
            flag1D = False
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
            import matplotlib.pyplot as plt

            plt.clf()
            fig = plt.figure()
            if flag1D:
                plt.plot(bin_x[:-1], F_all[-1])
                plt.xlabel("Projected CV", fontsize=14)
                plt.ylabel("Free energy [kJ/mol]", fontsize=14)
            else:
                pos = plt.imshow(
                    F_all[-1].T,
                    origin="lower",
                    interpolation="hanning",
                    extent=[bin_x[0], bin_x[-1], bin_y[0], bin_y[-1]],
                    aspect="auto",
                )
                plt.xlabel("CV 1", fontsize=14)
                plt.ylabel("CV 2", fontsize=14)
                cbar = plt.colorbar(pos)
                cbar.set_label("Free energy [kJ/mol]", rotation=270, labelpad=20)
                cbar.ax.tick_params(labelsize=8)
            plt.savefig("FitFES")
            plt.clf()

        if bias_factor is not None:
            return F_all[0], F_all[1], bin_x, bin_y
        return F, bin_x, bin_y

    def convergence_last_traj(
        self,
        dataset: CVfind,
        bias_factor,
        initial_height,
        batchsize=10000,
        bins=100,
        max_steps=1000000,
        noise=True,
    ):
        if self.net.bottleneck_dim > 2:
            print("Only possible with less than 3 CVs!")
            return
        splits = dataset.dataset.splits
        n_trajs = len(splits)
        if n_trajs > 1:
            weights = dataset.dataset.metad_weights
            # weights /= weights.sum() done in proj_cv
            new_cv_vals = self.estimate_cv_from_dataset(dataset, batchsize=batchsize)
            weights_full = np.concatenate(weights, axis=0)
            min_vals = new_cv_vals.min(0)
            max_vals = new_cv_vals.max(0)
            diff = (
                max_vals - min_vals
            )  # add 10% here as well as when estimating optimal features
            rel = diff / 10
            max_vals += rel
            min_vals -= rel
            if (
                dataset.dataset.bias_mask is not None
                and len(dataset.dataset.bias_mask) > 1
            ):
                frames_included = []
                start = 0
                for i, traj_dir in enumerate(dataset.dataset.traj_to_workdir):
                    end = dataset.total_length[i]
                    if traj_dir in dataset.dataset.bias_mask:
                        frames_included.append(np.arange(start, end))
                    start = end
                frames_included = np.concatenate(frames_included, axis=0)
            else:
                frames_included = None
            Fi, bin_x, bin_y = self.proj_cv(
                weights_full,
                new_cv_vals,
                bins,
                min_vals=min_vals,
                max_vals=max_vals,
                frames_included=frames_included,
            )

            F1_all = [Fi - Fi.mean()]
            if bias_factor is not None:
                from .util import fes_to_bias, fit_bias_gaussian, bias_to_fes

                assert (
                    initial_height is not None
                ), "If bias_factor is not None, initial_height must be given."
                bias_reconstruction = fes_to_bias(Fi, bias_factor)
                bias_new = fit_bias_gaussian(
                    bias_reconstruction,
                    initial_height=initial_height,
                    bias_factor=bias_factor,
                    lower=True,
                    max_steps=max_steps,
                    noise=noise,
                )
                F_temp = bias_to_fes(bias_new, bias_factor)
                F1_all.append(F_temp - F_temp.mean())

            # estimate on all but the last trajectory
            # estimate how many splits are in the last trajectory
            if (
                dataset.dataset.bias_mask is not None
                and len(dataset.dataset.bias_mask) > 1
            ):
                frames_included = []
                start = 0
                for i, traj_dir in enumerate(dataset.dataset.traj_to_workdir):
                    end = dataset.total_length[i]
                    if (
                        traj_dir in dataset.dataset.bias_mask[:-1]
                    ):  # remove the last trajectory
                        frames_included.append(np.arange(start, end))
                    start = end
                frames_included = np.concatenate(frames_included, axis=0)
            else:
                n_splits_last = len(splits[-1])
                if (
                    n_splits_last == 0
                ):  # if it is plain metadynamics, in case of segwalker it is different
                    n_splits_last = 1
                frames_but_last = dataset.total_length[-(n_splits_last + 1)]
                frames_included = np.arange(frames_but_last)
            Fj, bin_x, bin_y = self.proj_cv(
                weights_full,
                new_cv_vals,
                bins,
                min_vals=min_vals,
                max_vals=max_vals,
                frames_included=frames_included,
            )
            F2_all = [Fj - Fj.mean()]
            if bias_factor is not None:
                from .util import fes_to_bias, fit_bias_gaussian, bias_to_fes

                assert (
                    initial_height is not None
                ), "If bias_factor is not None, initial_height must be given."
                bias_reconstruction = fes_to_bias(Fj, bias_factor)
                bias_new = fit_bias_gaussian(
                    bias_reconstruction,
                    initial_height=initial_height,
                    bias_factor=bias_factor,
                    lower=True,
                    max_steps=max_steps,
                    noise=noise,
                )
                F_temp = bias_to_fes(bias_new, bias_factor)
                F2_all.append(F_temp - F_temp.mean())
            if bin_y is None:
                flag1D = True
            else:
                flag1D = False
            # estimate difference
            for counter, F1, F2 in zip(np.arange(len(F1_all)), F1_all, F2_all):
                diff = F1 - F2
                import matplotlib.pyplot as plt

                plt.clf()
                fig = plt.figure()
                if flag1D:
                    pos = plt.plot(bin_x[:-1], diff)
                    plt.ylabel("Free energy [kJ/mol]", fontsize=14)
                else:
                    pos = plt.imshow(
                        diff.T,
                        origin="lower",
                        interpolation="hanning",
                        extent=[bin_x[0], bin_x[-1], bin_y[0], bin_y[-1]],
                        aspect="auto",
                    )
                    plt.ylabel("CV 2", fontsize=14)
                    cbar = plt.colorbar(pos)
                    cbar.set_label("Free energy [kJ/mol]", rotation=270, labelpad=20)
                    cbar.ax.tick_params(labelsize=8)
                plt.xlabel("CV 1", fontsize=14)
                plt.savefig(f"Convergence{counter}")
                plt.clf()
                if counter == 1:
                    plt.clf()
                    fig = plt.figure()
                    if flag1D:
                        pos = plt.plot(bin_x[:-1], F1)
                        plt.ylabel("Free energy [kJ/mol]", fontsize=14)
                    else:
                        pos = plt.imshow(
                            F1.T,
                            origin="lower",
                            interpolation="hanning",
                            extent=[bin_x[0], bin_x[-1], bin_y[0], bin_y[-1]],
                            aspect="auto",
                        )
                        plt.ylabel("CV 2", fontsize=14)
                        cbar = plt.colorbar(pos)
                        cbar.set_label(
                            "Free energy [kJ/mol]", rotation=270, labelpad=20
                        )
                        cbar.ax.tick_params(labelsize=8)
                    plt.xlabel("CV 1", fontsize=14)

                    plt.savefig(f"Convergence_w_last")
                    plt.clf()
                    fig = plt.figure()
                    if flag1D:
                        pos = plt.plot(bin_x[:-1], F2)
                        plt.ylabel("Free energy [kJ/mol]", fontsize=14)
                    else:
                        pos = plt.imshow(
                            F2.T,
                            origin="lower",
                            interpolation="hanning",
                            extent=[bin_x[0], bin_x[-1], bin_y[0], bin_y[-1]],
                            aspect="auto",
                        )
                        plt.ylabel("CV 2", fontsize=14)
                        cbar = plt.colorbar(pos)
                        cbar.set_label(
                            "Free energy [kJ/mol]", rotation=270, labelpad=20
                        )
                        cbar.ax.tick_params(labelsize=8)
                    plt.xlabel("CV 1", fontsize=14)
                    plt.savefig(f"Convergence_wo_last")
        else:
            print("Only one trajectory, cannot estimate convergence.")


class CVEstimator_teacher(CVEstimator):
    r"""
    A class to find a surrogate model for the CVs in order to reduce computational costs for simulation.
    Parameters
    ----------
    net : CV_autoencoder
        The neural network to train.
    net_teacher : CV_autoencoder
        The teacher network, which is approximated by the simpler model.
    use_weights : bool, optional
        Whether to use weights from the FES for the training, by default True. Can give higher accuracy for metastable states.
    accelerator : str, optional
        The accelerator to use, by default 'gpu'
    devices : int, optional
        The number of devices to use, by default 1
    lr : float, optional
        The learning rate, by default 1e-3
    weight_decay : float, optional
        The weight decay, by default 1e-1
    """

    def __init__(
        self,
        net: CV_autoencoder,
        net_teacher: CV_autoencoder,
        use_weights: bool = True,
        accelerator="gpu",
        devices=1,
        lr=1e-3,
        weight_decay=1e-1,
        train_mask=False,
        precision_mode="16-mixed",
        vamp_loss=None,
        vamp_loss_weight=0,
        
    ):
        super().__init__(net, accelerator, devices, lr, weight_decay, vamp_loss=vamp_loss, vamp_loss_weight=vamp_loss_weight, precision_mode=precision_mode)
        self.net_teacher = net_teacher
        self.net.copy_teacher_to_net(self.net_teacher)
        self.net.switch_mask_training(train_mask)
        self.use_weights = use_weights
        self.validation_step_outputs = []
        self.validation_step_outputs_err = []

    def training_step(self, batch, batch_idx):
        r"""Performs a partial fit on data. This does not perform any batching.
        Parameters
        ----------
        batch : list
            The training data to feed through all the CVs, weights, and target data.
        batch_idx : int
            The index of the batch.

        Returns
        -------
        loss_value : torch.Tensor
        """
        x_t = []
        for x_t_i in batch[: self.n_cvs]:
            x_t.append(x_t_i)
        # x_tau = batch[-2]
        weights = batch[-1]
        if not self.use_weights:
            weights = torch.ones_like(weights)
        weights = weights / weights.sum()

        x_pred = self.net.bottleneck(x_t)
        x_teach = self.net_teacher.bottleneck(x_t)
        loss_value = loss_auto(x_pred, x_teach, weights)

        self.log("student_train_loss", loss_value)
        # self.log("train_score", loss_enc_item)
        return loss_value

    def validation_step(self, batch, batch_idx):
        r"""Performs validation on data. This does not perform any batching.

        Parameters
        ----------
        batch : list
            The validation data to feed through all the CVs, weights, and target data.
        batch_idx : int
            The index of the batch.

        Returns
        -------
        loss_value : np.ndarray
            The total loss value
        loss_enc_item : np.ndarray
            The loss value of the autoencoder
        err_item_list : List[np.ndarray]
            The loss value for the mask weights.

        """
        x_t = []
        for x_t_i in batch[: self.n_cvs]:
            x_t.append(x_t_i)
        # x_tau = batch[-2]
        weights = batch[-1]
        if not self.use_weights:
            weights = torch.ones_like(weights)
        weights = weights / weights.sum()

        x_pred = self.net.bottleneck(x_t)
        x_teach = self.net_teacher.bottleneck(x_t)
        loss_value = loss_auto(x_pred, x_teach, weights)
        self.validation_step_outputs.append(detach_numpy(loss_value))
        err_item_list = []
        for n_cv in range(self.n_cvs):
            err = self.net.weights_norm(n_cv)
            err_item = detach_numpy(err)
            err_item_list.append(err_item)
        self.validation_step_outputs_err.append(err_item_list)
        return detach_numpy(loss_value), err_item_list

    def on_validation_epoch_end(self):
        loss_mean = np.mean(self.validation_step_outputs)
        err_item = np.mean(self.validation_step_outputs_err, axis=0)
        for i in range(len(err_item)):
            self.log(f"weight_norm{i+1}", err_item[i])
        self.log("student_val_loss", loss_mean)
        self.validation_step_outputs_err.clear()
        self.validation_step_outputs.clear()
        return loss_mean

    def configure_optimizers(self):
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        return self.optimizer

    def fit_routine(
        self, data_loader, validation_loader, patience=2, max_steps=1000000
    ):
        """Fitting routine for estimating the CVs of the complex model with a simple model.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            The loader for training data.
        validation_loader : torch.utils.data.DataLoader
            The loader for validation data used for the early stopping.
        patience : int, optional
            Patience used for the early stopping, by default 2
        """
        self.trainer = pl.Trainer(
            accelerator=self.accelerator,
            devices=self.devices,
            max_steps=max_steps,
            precision=self.precision_mode,
        )
        self.trainer.callbacks[0].disable()  # disable progress bar
        self.trainer.callbacks.pop(2)  # remove modelckeckpoint
        self.trainer.callbacks += [
            EarlyStopping(monitor="student_val_loss", mode="min", patience=patience)
        ]
        self.trainer.fit(self, data_loader, validation_loader)
        self.save("./model_student.pt")
        self.log_active_features(data_loader.dataset.dataset.dataset)
        # plot difference between two models
