# create a model accessible from the command line

from addict import Dict
from timelaggedcv.dataset import CVfind, CV_timelagged, create_config_pickle
from . import CV_autoencoder
from . import CVEstimator, CVEstimator_teacher
from torch import cuda
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import numpy as np


class TimeLaggedCV(object):
    DEFAULT = Dict(
        analysis={
            "safety_percent": 0.1,  # amount to add to min/max of CVs
            "bias_factor": 15.,
            "bins": 20,
            "initial_height": 1.,
        },
        create_dataset_options={
            "pickle_descriptor": None,
            "pickle_features": None,
            "skip": 1,
            "save_path": "../features/",
            "stride": 1,
            "workdirs": None,
            "time_mode": "bias",  # c_t, bias, or diff
            "max_time": 10000,  # the largest possible fake time step within a frame, the others are scaled accordingly
        },
        dataset={
            "tau": 250,
        },
        general={
            "batch_size": 1024,
            "lr": 0.001,
            "lr_decoder": None,
            "teacher": True,  # whether to use teacher and student model
            "student_depth": 2,
            "student_width": 3,
            "student_train_mask": True,
            "use_weights_student": False,
            "valid_ratio": 0.2,
            "weight_decay": 0.1,
            "num_workers": 0,
            "prefetch_factor": 2,
            "accelerator": None,
            "work_dir": "./",
            "finetune": True,
            "precision_mode": "16-mixed", # 16-mixed or 32-true
            "weighting_features": True, # weight the features connected to the cvs
            "vamp_loss": "VAMP2", # VAMP2, VAMP1, VAMPE or None
            "vamp_loss_weight": 1.0,
        },
        nn={
            "cv_acti": "Tanh",
            "cv_depth": 4,  # if teacher, this is the size of the CV model
            "cv_width": 50,
            "n_res_blocks": 6,  # decoder architecture
            "residual": False,
            "width": 200,
            "score_method": "L2",
            "special_cv": True,  # more complex architecture for the teacher model
            "special_residual": True,
            "special_rewiring": True,
            "special_skip_res": 1,
            "sigma_bin": 5,
            "sigma_feat_mask":1.,
        },
        opt={
            "load_model": False,  # load the converged model without constraints
            "patience": 2,  # number of rounds to wait before considered converged
            "step_lam": 0.2,  # the step size to increase the penalty
            "target_n_feat": 10,  # number of features to be used in the CV model
            "updates_per_cycle": 300,
            "max_steps": 10_000,
            "max_rounds": 3,
        },
    )

    def __init__(self, options={}):
        self.options = self.get_options(options)
        print(f"RSMLCV using options: {self.options.to_dict()}")
        if self.options.general["num_workers"] == 0:
            self.options.general["prefetch_factor"] = None
        # create dataset

        if self.options.general['vamp_loss'] is not None: # for linalg needs 32 precision
            self.options.general['precision_mode'] = '32-true'
        create_config_pickle(self.options.create_dataset_options['pickle_descriptor'],
                                self.options.create_dataset_options['pickle_features'],
                                self.options.create_dataset_options['skip'])
        self.dataset = CV_timelagged(self.options.dataset['tau'])
        print("Number of data pairs: {}".format(len(self.dataset)))
        self.dataset_weight = CVfind(self.dataset)
        # create CV networks
        self.options.nn["input_dims"] = self.dataset.input_dims
        self.options.nn["output_dim"] = self.dataset.output_dim
        self.options.nn["n_bins"] = self.options.analysis["bins"] * 5
        self.options.nn["feat_indexes_input"] = self.dataset.feat_indexes_input
        print("output_dim: {}".format(self.options.nn["output_dim"]))
        n_output_feat = len(self.dataset.feat_indexes_output)
        weights_features = np.ones(n_output_feat)
        if self.options.general['weighting_features']:
            for input_feat in self.dataset.feat_indexes_input:
                n_input_feat = len(input_feat)
                ratio = n_input_feat/n_output_feat
                weights_features[input_feat] /= ratio
            # sum again to number of features
            weights_features*=len(weights_features)/weights_features.sum()
        self.options.nn['weights_feat'] = weights_features
        self.options.nn['vamp_loss'] = self.options.general['vamp_loss']
        self.net = CV_autoencoder(**self.options.nn)
        # set mean and std of dataset
        self.net.set_mean_std(self.dataset.means, self.dataset.stds)
        # create estimator
        if self.options.general["accelerator"] is None:
            self.accelerator = "gpu" if cuda.is_available() else "cpu"
        else:
            self.accelerator = self.options.general["accelerator"]
        self.devices = 1
        self.cvEstimator = CVEstimator(
            self.net,
            self.accelerator,
            self.devices,
            self.options.general["lr"],
            self.options.general["lr_decoder"],
            self.options.general["weight_decay"],
            self.options.general['vamp_loss'],
            self.options.general["vamp_loss_weight"],
            precision_mode=self.options.general["precision_mode"],
        )

        # if teacher student model, create student network
        self.teacher_flag = self.options.general["teacher"]
        if self.teacher_flag:
            from copy import deepcopy

            student_nn_params = deepcopy(self.options.nn)
            student_nn_params["cv_depth"] = self.options.general["student_depth"]
            student_nn_params["cv_width"] = self.options.general["student_width"]
            student_nn_params["decoder"] = False
            student_nn_params["special_cv"] = False
            student_nn_params["sigma_feat_mask"] = 0.
            self.student_net = CV_autoencoder(**student_nn_params)
            self.student_cvEstimator = CVEstimator_teacher(
                self.student_net,
                self.net,
                self.options.general["use_weights_student"],
                self.accelerator,
                self.devices,
                self.options.general["lr"],
                weight_decay=0.0,
                train_mask=self.options.general["student_train_mask"],
                vamp_loss=self.options.general['vamp_loss'],
                vamp_loss_weight=self.options.general["vamp_loss_weight"],
                precision_mode=self.options.general["precision_mode"],
            )
            self.dataset_student = CVfind(self.dataset, True)
            print("Number of data pairs student: {}".format(len(self.dataset_student)))

    @classmethod
    def get_options(cls, options={}):
        combined_options = Dict(cls.DEFAULT)
        combined_options.update(Dict(options))
        return combined_options

    def reset_networks(self):
        self.net = CV_autoencoder(**self.options.nn)
        self.cvEstimator = CVEstimator(
            self.net,
            self.accelerator,
            self.devices,
            self.options.general["lr"],
            self.options.general["lr_decoder"],
            self.options.general["weight_decay"],
        )
        if self.teacher_flag:
            from copy import deepcopy

            student_nn_params = deepcopy(self.options.nn)
            student_nn_params["cv_depth"] = self.options.general["student_depth"]
            student_nn_params["cv_width"] = self.options.general["student_width"]
            student_nn_params["decoder"] = False
            self.student_net = CV_autoencoder(**student_nn_params)
            self.student_cvEstimator = CVEstimator_teacher(
                self.student_net,
                self.net,
                self.options.general["use_weights_student"],
                self.accelerator,
                self.devices,
                self.options.general["lr"],
                weight_decay=0.0,
                train_mask=self.options.general["student_train_mask"],
            )

    def create_training_sets(self):
        n_val = int(len(self.dataset) * self.options.general["valid_ratio"])

        train_data, val_data = random_split(
            self.dataset, [len(self.dataset) - n_val, n_val]
        )
        if self.options.general["num_workers"] == 0:
            loader_train = DataLoader(
                train_data, batch_size=self.options.general["batch_size"], shuffle=True
            )
        else:
            loader_train = DataLoader(
                train_data,
                batch_size=self.options.general["batch_size"],
                shuffle=True,
                num_workers=self.options.general["num_workers"],
                prefetch_factor=self.options.general["prefetch_factor"],
            )
        loader_val = DataLoader(
            val_data, batch_size=self.options.general["batch_size"] * 2, shuffle=False
        )
        if len(train_data) < self.options.general["batch_size"]:
            n_batches_train = 1
        else:
            n_batches_train = len(train_data) // self.options.general["batch_size"]
        waiting_rounds = np.ceil(
            self.options.opt["updates_per_cycle"] / n_batches_train
        ).astype(
            int
        )  # epochs per round
        if waiting_rounds < 1:
            waiting_rounds = 1
        self.options.opt["waiting_rounds"] = int(waiting_rounds)

        return loader_train, loader_val

    def create_training_sets_student(self):
        n_val = int(len(self.dataset_student) * self.options.general["valid_ratio"])
        # n_test = int(len(dataset)*test_ratio)
        train_data, val_data = random_split(
            self.dataset_student, [len(self.dataset_student) - n_val, n_val]
        )
        if self.options.general["num_workers"] == 0:
            loader_train_app = DataLoader(
                train_data, batch_size=self.options.general["batch_size"], shuffle=True
            )
        else:
            loader_train_app = DataLoader(
                train_data,
                batch_size=self.options.general["batch_size"],
                shuffle=True,
                num_workers=self.options.general["num_workers"],
                prefetch_factor=self.options.general["prefetch_factor"],
            )
        loader_val_app = DataLoader(
            val_data, batch_size=self.options.general["batch_size"] * 2, shuffle=False
        )

        return loader_train_app, loader_val_app

    def train(self, loader_train, loader_val):
        opt_params = self.options.opt
        # train model
        self.cvEstimator.fit_routine(
            opt_params["target_n_feat"],
            loader_train,
            loader_val,
            opt_params["patience"],
            opt_params["step_lam"],
            load_model=opt_params["load_model"],
            max_steps=opt_params["max_steps"],
            max_rounds=opt_params["max_rounds"],
        )

    def train_student(self, loader_train, loader_val):
        opt_params = self.options.opt
        self.student_cvEstimator.fit_routine(
            loader_train,
            loader_val,
            opt_params["patience"],
            max_steps=opt_params["max_steps"],
        )  # should patience be larger?

    def predict(self, work_dir="./"):
        model = self.get_model()
        model.get_optimal_feats(
            work_dir,
            self.dataset_weight,
            safety_percent=self.options.analysis["safety_percent"],
        )
        model.estimate_fes_cv(
            self.dataset_weight,
            work_dir,
            bins=self.options.analysis["bins"] * 5,
            bias_factor=self.options.analysis["bias_factor"],
            initial_height=self.options.analysis["initial_height"],
            safety_percent=self.options.analysis["safety_percent"],
        )

    def test_convergence(self):
        model = self.get_model()
        model.convergence_last_traj(
            self.dataset_weight,
            bias_factor=self.options.analysis["bias_factor"],
            bins=self.options.analysis["bins"] * 5,
            initial_height=self.options.analysis["initial_height"],
        )

    def get_model(self):
        if self.teacher_flag:
            model = self.student_cvEstimator
        else:
            model = self.cvEstimator
        return model

    def run(self):
        # create training sets
        loader_train, loader_val = self.create_training_sets()
        # train model
        self.train(loader_train, loader_val)
        self.net.switch_sigma_feat_mask(0)
        # if teacher student model, create student training sets and train student model
        if self.teacher_flag:
            self.cvEstimator.estimate_fes_cv(
                self.dataset_weight,
                self.options.general["work_dir"],
                bins=self.options.analysis["bins"] * 5,
                bias_factor=self.options.analysis["bias_factor"],
                initial_height=self.options.analysis["initial_height"],
                safety_percent=self.options.analysis["safety_percent"],
            )
            self.student_net.copy_teacher_to_net(self.net)
            print("active are: {}".format(self.student_net.get_active(0)))
            loader_train_app, loader_val_app = self.create_training_sets_student()
            self.train_student(loader_train_app, loader_val_app)
            if self.options.general["finetune"]:
                self.predict(
                    self.options.general["work_dir"])
                self.cvEstimator.fit_finetune_student(
                    self.student_net,
                    loader_train,
                    loader_val,
                    self.options.opt["patience"],
                )
        # predict
        self.predict(
            self.options.general["work_dir"],
        )
        # test convergence
        self.test_convergence()
