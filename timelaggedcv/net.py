# stores all the neural network classes to estimate CVs
import torch
import numpy as np


from .util import dump_yaml
import warnings
from copy import deepcopy
from .util import loss_auto

from timelaggedcv.vamp import vampnet_loss

class FastSoftmax(torch.nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        x = torch.abs(x * self.alpha)
        x = x / torch.sum(x, dim=-1, keepdim=True)
        return x


class CV_NN(torch.nn.Module):
    """Neural network CV, which can be used to construct the CV autoencoder.

    Parameters
    ----------
    input_dim : int
        The dimension of the input features.
    width : int, optional
        The width of the bottleneck layer, by default 2
    depths : int, optional
        The number of layers in the bottleneck, by default 1
    acti : str, optional
        The activation function to use in the bottleneck, by default 'Tanh'
    """

    def __init__(self, input_dim, width=2, depths=1, acti="Tanh", mean=None, std=None, sigma_feat_mask=0.):
        super().__init__()
        # first parameters for mapping to bottleneck
        self.input_dim = input_dim
        self.bottleneck_dim = 1
        # self.threshold = torch.nn.Parameter(data=torch.zeros(1), requires_grad=False)
        self.kernel_bottleneck = torch.nn.Parameter(
            data=torch.ones((input_dim)), requires_grad=True
        )  # think about removing the division!
        self.bottleneck_width = width
        self.bott_depths = depths
        self.sigma_feat_mask = torch.nn.Parameter(torch.tensor(sigma_feat_mask), requires_grad=False)
        self.bottleneck_layer = (
            [torch.nn.Linear(input_dim, self.bottleneck_width)]
            + [
                torch.nn.Linear(self.bottleneck_width, self.bottleneck_width)
                for _ in range(self.bott_depths - 1)
            ]
            + [torch.nn.Linear(self.bottleneck_width, self.bottleneck_dim, bias=False)]
        )
        self.params_bottleneck = torch.nn.ModuleList(self.bottleneck_layer)
        self.bottleneck_acti = acti
        self.whitening_layer = Mean_std_layer(input_dim, mean, std)
        if acti == "Exp":
            self.bott_acti = (
                torch.exp
            )  # torch.nn.ELU()#torch.exp#torch.nn.Softplus()#torch.nn.Tanh()
        elif acti == "ELU":
            self.bott_acti = torch.nn.ELU()
        elif acti == "Tanh":
            self.bott_acti = torch.nn.Tanh()
        elif acti == "Softplus":
            self.bott_acti = torch.nn.Softplus()
        else:
            self.bott_acti = torch.nn.Identity()
        self.acti_mask = torch.nn.ReLU()

    def estimate_masked_kernel(self):
        """Estimates the masked kernel, which is the kernel multiplied by the mask. The threshold can create a gap between positive and negative values. Thereby,
        weights can be pushed to zero."""
        masked_kernel = self.acti_mask(self.kernel_bottleneck)
        return masked_kernel

    def forward(self, features):
        # first apply mask
        features = self.whitening_layer(features)
        kernel_masked = self.estimate_masked_kernel()
        feat = features * kernel_masked[None, :]
        if self.sigma_feat_mask>0:
            feat += torch.randn_like(features) * self.sigma_feat_mask * self.acti_mask(1-kernel_masked[None,:])
        # expand to the right number of nodes
        for layer in self.bottleneck_layer[:-1]:
            feat = self.bott_acti(layer(feat))
        ret = self.bottleneck_layer[-1](feat)
        return ret

    def weights_norm(self):
        """Estimates the specified norm on the weights still active according to the mask over all the features.

        Parameters
        ----------
        norm_type : str, optional
            Either L1 norm or L2, by default 'L1'
        Returns
        -------
        float
            The norm value of the active features.
        """
        norm = torch.mean(torch.abs(self.estimate_masked_kernel()))
        return norm

    def get_active(self):
        """Estimates the number of active features, i.e. the number of features with a non-zero weight."""
        kernel_dim = self.estimate_masked_kernel()
        n_positive = (kernel_dim > 0).sum().item()
        return int(n_positive)

    def set_mean_std(self, mean, std):
        """Sets the mean and standard deviation for the weightening layer.

        Parameters
        ----------
        mean : torch.Tensor
            The mean of the features.
        std : torch.Tensor
            The standard deviation of the features.
        """
        self.whitening_layer.set_both(mean, std)

    def get_opt_feat(self, workdir, dataset, max_val, min_val, n_cv):
        """Estimates the features still active according to the mask and saves them to a file. Additionally, it saves
        all the parameters necessary (not the ones from inactive features). The file can be used to construct the CV within openmm.

        Parameters
        ----------
        workdir : str
            The working directory, where the file will be saved.
        dataset : rsmlcv.dataset.Dataset
            The dataset object, which describes the meaning of the different features.
        max_val : float
            The maximum value of the features.
        min_val : float
            The minimum value of the features.
        n_cv : int
            The number of the CV in the dataset, which should be saved.
        """
        list_cv, weights_list = self.get_feature_weights(dataset, n_cv)
        cv_max = [max_val]

        cv_min = [min_val]

        final_file = {
            "weights_feat": list_cv,
            "weights_layer": weights_list,
            "cv_max": cv_max,
            "cv_min": cv_min,
            "acti": self.bottleneck_acti,
        }
        dump_yaml(final_file, workdir + f"CV_file{n_cv}.yaml")

    def get_feature_weights(self, dataset, n_cv):
        values = self.get_kernel()
        sort_id = np.argsort(np.abs(values))
        sort_n = sort_id[::-1]
        active = np.abs(values[sort_n]) > 0.0
        feat_ind = sort_n[active]
        param_kernel = values[sort_n[active]].astype("float64")
        weights_list = []
        for layer_i, layer in enumerate(self.bottleneck_layer):
            params_temp = [
                param.detach().cpu().numpy().astype("float64")
                for param in layer.parameters()
            ]
            W = params_temp[0].T
            if layer_i == 0:
                W = W[feat_ind]  # only save the relevant ones in the right order!
            if len(params_temp) == 2:
                weights_list.append({"W": W, "b": params_temp[1]})
            else:
                weights_list.append({"W": W})
        list_cv = []
        for feat_i, param_i in zip(feat_ind, param_kernel):
            dict_temp = deepcopy(dataset.feat_nn_list[n_cv][feat_i])
            dict_temp["parameter"] = param_i
            list_cv.append(dict_temp)
        return list_cv, weights_list

    def get_kernel(self):
        return self.estimate_masked_kernel().detach().cpu().numpy()


class CV_NN_special(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        width=50,
        acti=torch.nn.Tanh(),
        acti_sm=FastSoftmax(alpha=1.0),
        residual=True,
        n_res_blocks=4,
        res_skip=1,
        mean=None,
        std=None,
        rewiring=True,
        final_bias=False,
        sigma_feat_mask=0.,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = 1
        # self.threshold = torch.nn.Parameter(data=torch.zeros(1), requires_grad=False)
        self.kernel_bottleneck = torch.nn.Parameter(
            data=torch.ones((input_dim)), requires_grad=True
        )  # think about removing the division!
        self.bottleneck_width = width
        self.residual = residual
        self.bottleneck_acti = acti
        self.whitening_layer = Mean_std_layer(input_dim, mean, std)
        self.acti_mask = torch.nn.ReLU()
        self.sigma_feat_mask = torch.nn.Parameter(torch.tensor(sigma_feat_mask), requires_grad=False)
        self.acti = acti
        self.rewiring = rewiring
        self.final_acti = torch.nn.Identity()  # torch.nn.Tanh()
        if rewiring:
            self.first_layer = SelfAttentionRewire(
                input_dim, input_dim, width, acti_sm=acti_sm, acti_v=acti, acti=acti
            )
            self.res_blocks = torch.nn.ModuleList(
                [
                    torch.nn.ModuleList(
                        [
                            SelfAttentionRewire(
                                input_dim,
                                width,
                                width,
                                acti_sm=acti_sm,
                                acti_v=acti,
                                acti=acti,
                            )
                            for _ in range(res_skip)
                        ]
                    )
                    for _ in range(n_res_blocks)
                ]
            )
        else:
            self.first_layer = SelfAttention(
                input_dim, width, acti_sm=acti_sm, acti_v=acti, acti=acti
            )
            self.res_blocks = torch.nn.ModuleList(
                [
                    torch.nn.ModuleList(
                        [
                            SelfAttention(
                                width,
                                width,
                                acti_sm=acti_sm,
                                acti=acti,
                                acti_v=acti,
                            )
                            for _ in range(res_skip)
                        ]
                    )
                    for _ in range(n_res_blocks)
                ]
            )
        self.output_layer = torch.nn.Linear(width, self.bottleneck_dim, bias=final_bias)

    def forward(self, features):
        # first apply mask
        features = self.whitening_layer(features)
        kernel_masked = self.estimate_masked_kernel()
        feat = features * kernel_masked[None, :]
        if self.sigma_feat_mask>0:
            feat += torch.randn_like(features) * self.sigma_feat_mask * self.acti_mask(1-kernel_masked[None,:])
        if self.rewiring:
            x = self.acti(self.first_layer(feat, feat))
        else:
            x = self.acti(self.first_layer(feat))
        for res_block in self.res_blocks:
            y = x
            for layer in res_block:
                if self.rewiring:
                    y = self.acti(layer(feat, y))
                else:
                    y = self.acti(layer(y))
            if self.residual:
                x = x + y
            else:
                x = y
        ret = self.output_layer(x)
        return ret

    def estimate_masked_kernel(self):
        """Estimates the masked kernel, which is the kernel multiplied by the mask. The threshold can create a gap between positive and negative values. Thereby,
        weights can be pushed to zero."""
        masked_kernel = self.acti_mask(self.kernel_bottleneck)
        return masked_kernel

    def weights_norm(self):
        """Estimates the specified norm on the weights still active according to the mask over all the features.

        Parameters
        ----------
        norm_type : str, optional
            Either L1 norm or L2, by default 'L1'
        Returns
        -------
        float
            The norm value of the active features.
        """
        norm = torch.mean(torch.abs(self.estimate_masked_kernel()))
        return norm

    def get_active(self):
        """Estimates the number of active features, i.e. the number of features with a non-zero weight."""
        kernel_dim = self.estimate_masked_kernel()
        n_positive = (kernel_dim > 0).sum().item()
        return int(n_positive)

    def set_mean_std(self, mean, std):
        """Sets the mean and standard deviation for the weightening layer.

        Parameters
        ----------
        mean : torch.Tensor
            The mean of the features.
        std : torch.Tensor
            The standard deviation of the features.
        """
        self.whitening_layer.set_both(mean, std)

    def get_kernel(self):
        return self.estimate_masked_kernel().detach().cpu().numpy()

    def get_opt_feat(self, workdir, dataset, max_val, min_val, n_cv):
        warnings.warn(
            "Not implemented yet! Special CV has no features to save, because it is too complex to be used with enhanced sampling"
        )

class CV_decoder_features(torch.nn.Module):
    """The decoder part of the auto-encoder model. It tries to map the CV space to the time-lagged feature space.

    Parameters
    ----------
    bottleneck_dim : int
        The dimension of the CV space.
    output_dim : int
        The dimension of the time-lagged feature space.
    width : int, optional
        The width of the hidden layers, by default 100
    residual : bool, optional
        Whether to use residual connections, by default True
    n_res_blocks : int, optional
        The number of residual blocks, by default 4
    res_skip : int, optional
        Number of layers within a residual block, by default 1
    acti : torch.nn.Module, optional
        The activation function, by default torch.nn.ELU()
    """

    def __init__(
        self,
        bottleneck_dim,
        output_dim,
        width=100,
        residual=True,
        n_res_blocks=4,
        res_skip=1,
        acti=torch.nn.ELU(),
        rewiring=False,
        acti_sm=FastSoftmax(alpha=1.0),
        sigma_bin=None,
        n_bins=100,
    ):
        super().__init__()
        # first parameters for mapping to bottleneck
        self.residual = residual
        self.rewiring = rewiring
        if acti is None:
            self.acti = torch.nn.Identity()
        else:
            self.acti = acti
        if rewiring:
            self.input_layer = SelfAttentionRewire(
                bottleneck_dim, bottleneck_dim, width, acti_sm=acti_sm, acti_v=acti, acti=acti
            )
            self.res_blocks = torch.nn.ModuleList(
                [
                    torch.nn.ModuleList(
                        [
                            SelfAttentionRewire(
                                bottleneck_dim,
                                width,
                                width,
                                acti_sm=acti_sm,
                                acti_v=acti,
                                acti=acti,
                            )
                            for _ in range(res_skip)
                        ]
                    )
                    for _ in range(n_res_blocks)
                ]
            )
        else:
            self.input_layer = torch.nn.Linear(bottleneck_dim, width)
            self.res_blocks = torch.nn.ModuleList(
                [
                    torch.nn.ModuleList(
                        [torch.nn.Linear(width, width) for _ in range(res_skip)]
                    )
                    for _ in range(n_res_blocks)
                ]
            )
        self.output_layer = torch.nn.Linear(width, output_dim)
        self.sigma_bin = sigma_bin
        self.n_bins = n_bins

    def forward(self, cv_features, noise=True):
        """Forward pass of the decoder mapping the CV space to the time-lagged feature space."""
        feat = cv_features
        if self.sigma_bin is not None and noise:
            cv_max = torch.max(feat, dim=0, keepdim=True)[0].detach()
            cv_min = torch.min(feat, dim=0, keepdim=True)[0].detach()
            diff = (cv_max - cv_min) * 1.2  # safety factor
            sigma = diff * self.sigma_bin / self.n_bins
            feat = feat + torch.randn_like(feat) * sigma
        if self.rewiring:
            x = self.acti(self.input_layer(feat, feat))
        else:
            x = self.acti(self.input_layer(feat))
        for res_block in self.res_blocks:
            y = x
            for layer in res_block:
                if self.rewiring:
                    y = self.acti(layer(feat, y))
                else:
                    y = self.acti(layer(y))
            if self.residual:
                x = x + y
            else:
                x = y
        x = self.output_layer(x)
        return x

class CV_autoencoder(torch.nn.Module):
    """The auto-encoder model. It tries to map the feature space to the CV space and back to the time-lagged feature space.
    It can be constructed of multiple CV_NN and CV_fixed models, which are concatenated to the CV space. The decoder part is a CV_decoder model.

    Parameters
    ----------
    input_dims_nn : list
        The dimensions of the feature spaces of the CV_NN models.
    output_dim : int
        The dimension of the time-lagged feature space.
    CV_types : list, optional
        The types of the CV models, which gives the total number of CVs, by default ['all', 'all']
        If CV_types is longer than the length of input_dims_nn, the rest will be CV_fixed models.
    cv_width : int, optional
        The width of the hidden layers of the CV_NN models, by default 2
    cv_depths : int, optional
        The number of hidden layers of the CV_NN models, by default 1
    cv_acti : str, optional
        The activation function of the CV_NN models, by default 'Tanh'
    width : int, optional
        The width of the hidden layers of the CV_decoder model, by default 100
    residual : bool, optional
        Whether to use residual connections in the CV_decoder model, by default True
    n_res_blocks : int, optional
        The number of residual blocks in the CV_decoder model, by default 4
    res_skip : int, optional
        Number of layers within a residual block in the CV_decoder model, by default 1
    acti : torch.nn.Module, optional
        The activation function of the CV_decoder model, by default torch.nn.ELU()
    decoder : bool, optional
        Whether to use a decoder, by default True
    """

    def __init__(
        self,
        input_dims: list,
        output_dim: int,
        cv_width=2,
        cv_depth=1,
        cv_acti="Tanh",
        means=None,
        stds=None,
        width=100,
        residual=True,
        n_res_blocks=4,
        res_skip=1,
        acti=torch.nn.ELU(),
        decoder=True,
        score_method="L2",
        special_cv=False,
        special_residual=True,
        special_rewiring=False,
        special_skip_res=1,
        sigma_bin=None,
        n_bins=100,
        feat_indexes_input=None,
        weights_feat=None,
        vamp_loss=None,
        sigma_feat_mask=0.,
    ) -> None:
        super().__init__()
        cv_nets = []
        counter_nn = 0
        self.index_nn_cv = []
        self.output_dim = output_dim
        self.cv_nets_inputs = []
        self.sigma_feat_mask=sigma_feat_mask
        self.feat_indexes_input = feat_indexes_input
        for i, input_dim in enumerate(input_dims):
            if means is None:
                mean, std = None, None
            else:
                mean, std = means[counter_nn], stds[counter_nn]
                counter_nn += 1
            if special_cv:
                cv_nets.append(
                    CV_NN_special(
                        input_dim,
                        cv_width,
                        acti=torch.nn.Tanh(),
                        residual=special_residual,
                        rewiring=special_rewiring,
                        n_res_blocks=cv_depth,
                        res_skip=special_skip_res,
                        mean=mean,
                        std=std,
                        sigma_feat_mask=sigma_feat_mask,
                    )
                )
            else:
                cv_nets.append(
                    CV_NN(input_dim, cv_width, cv_depth, cv_acti, mean, std, sigma_feat_mask=sigma_feat_mask)
                )
            self.index_nn_cv.append(i)
            self.cv_nets_inputs.append(input_dim)
        self.n_cv_nn = len(self.index_nn_cv)
        self.bottleneck_dim = len(cv_nets)
        self.cv_nets = torch.nn.ModuleList(cv_nets)
        self.score_method = score_method
        self.weights_feat = torch.Tensor(weights_feat) if weights_feat is not None else weights_feat
        self.vamp_loss = vamp_loss
        if decoder:
            self.whitening_layer = Mean_std_layer(output_dim)
            
            self.decoder = CV_decoder_features(
                self.bottleneck_dim,
                output_dim,
                width,
                residual,
                n_res_blocks,
                res_skip,
                acti,
                rewiring=special_rewiring,
                sigma_bin=sigma_bin,
                n_bins=n_bins,
            )
        else:
            self.decoder = torch.nn.Identity()
            self.whitening_layer = torch.nn.Identity()
        self.relu_func = torch.nn.ReLU()

    def bottleneck(self, inputs_list):
        """Forward pass of the CV models, which maps the feature space to the CV space."""
        cv_list = []
        for (
            cv_net,
            inputs_i,
        ) in zip(self.cv_nets, inputs_list):
            cv_list.append(cv_net.forward(inputs_i))
        bottleneck = torch.cat(cv_list, axis=-1)
        return bottleneck

    def forward(self, inputs_list):
        bottleneck = self.bottleneck(inputs_list)
        decoded = self.decoder.forward(bottleneck)
        return decoded, bottleneck

    def weights_norm(self, n_cv):
        """Estimates the specified norm on the weights still active according to the mask over all the features for the specified CV.

        Parameters
        ----------
        n_cv : int
            The index of the CV.
        norm_type : str, optional
            The norm type, by default 'L1'
        Returns
        -------
        float
            The norm value for the specified CV.
        """
        assert (
            n_cv <= self.n_cv_nn
        ), "There exist only {} many NN CVs, but you asked for {}".format(
            self.n_cv_nn, n_cv
        )
        return self.cv_nets[n_cv].weights_norm()

    def estimate_auto_loss(self, batch):
        """Estimates the auto-encoder loss for the specified batch."""
        x_t = []
        for x_t_i in batch[:self.bottleneck_dim]:
            x_t.append(x_t_i)
        x_tau = batch[-2]
        weights = batch[-1]
        weights = weights / weights.sum()

        x_pred, bn = self.forward(x_t)
        
        if self.vamp_loss:
            # get the x_tau_input features
            x_tau_input = []
            for _, indexes in enumerate(self.feat_indexes_input):
                x_tau_input.append(x_tau[:,indexes])
            bn_tau = self.bottleneck(x_tau_input)
            vamp_loss = vampnet_loss(bn, bn_tau, weights, sym=True,
                                     method=self.vamp_loss, epsilon=1e-6, mode='trunc') # remove given CVs!
        else:
            vamp_loss = 0
        # estimate deviations from range -1, 1 of bottleneck
        bn_outer = self.relu_func((torch.abs(bn) - 1))
        x_tau = self.whitening_layer(x_tau)
        if self.weights_feat is not None:
            if x_tau.device != self.weights_feat.device:
                self.weights_feat = self.weights_feat.to(x_tau.device)
        loss_enc = loss_auto(x_pred, x_tau, weights, norm=self.score_method, weights_feat=self.weights_feat)
        loss_outer = (bn_outer**2).mean(0)
        return loss_enc, loss_outer, vamp_loss

    def get_active(self, n_cv):
        """Estimates the number of active features for the specified CV."""
        assert (
            n_cv <= self.n_cv_nn
        ), "There exist only {} many NN CVs, but you asked for {}".format(
            self.n_cv_nn, n_cv
        )
        kernel_dim = self.cv_nets[n_cv].estimate_masked_kernel()
        n_positive = (kernel_dim > 0).sum().item()
        return int(n_positive)

    def get_opt_feat(self, workdir, dataset, max_vals, min_vals):
        """Estimates the active features for the specified CVs and the corresonding network parameters."""
        max_vals = max_vals.astype("float64")
        min_vals = min_vals.astype("float64")
        for n_cv in range(self.n_cv_nn):
            self.cv_nets[n_cv].get_opt_feat(
                workdir, dataset, max_vals[n_cv], min_vals[n_cv], n_cv
            )

    def switch_mask_training(self, requires_grad=True):
        """Switches the mask training on or off."""
        for n_cv in range(self.n_cv_nn):
            self.cv_nets[n_cv].kernel_bottleneck.requires_grad = requires_grad
            if not requires_grad:
                self.cv_nets[n_cv].kernel_bottleneck.grad = None
            else:
                self.cv_nets[n_cv].kernel_bottleneck.grad = torch.zeros_like(
                    self.cv_nets[n_cv].kernel_bottleneck
                )

    def switch_individual_mask_training(self, n_cv, requires_grad=True):
        """Switches the mask training on or off for the specified CV."""
        self.cv_nets[n_cv].kernel_bottleneck.requires_grad = requires_grad
        if not requires_grad:
            self.cv_nets[n_cv].kernel_bottleneck.grad = None
        else:
            self.cv_nets[n_cv].kernel_bottleneck.grad = torch.zeros_like(
                self.cv_nets[n_cv].kernel_bottleneck
            )

    def switch_sigma_feat_mask(self, sigma_new=0):
        """Changes the noise added to the masked features. Important for final prediction"""
        for n_cv in range(self.n_cv_nn):
            with torch.no_grad():
                self.cv_nets[n_cv].sigma_feat_mask.copy_(torch.tensor(sigma_new))

    def set_mean_std(self, means, stds):
        """Sets the mean and std for the specified CVs."""
        for n_cv in self.index_nn_cv:
            self.cv_nets[n_cv].set_mean_std(means[n_cv], stds[n_cv])
        if len(means) > self.n_cv_nn:
            self.whitening_layer.set_both(means[-1], stds[-1])

    def get_input_dim(self, n_cv):
        """Estimates the input dimension for the specified CV."""
        return self.cv_nets[n_cv].input_dim

    def get_kernel(self, n_cv):
        return self.cv_nets[n_cv].get_kernel()

    def get_perc(self, n_cv):
        n_active = self.get_active(n_cv)
        return n_active / self.get_input_dim(n_cv)

    def copy_teacher_to_net(self, teacher):
        """Copies the mask parameters from the teacher to self."""
        for cv_n in range(self.n_cv_nn):
            with torch.no_grad():
                # get current value
                kernel_cur = deepcopy(teacher.cv_nets[cv_n].kernel_bottleneck)
                # estimate the factor for which it will change due to the new threshol
                self.cv_nets[cv_n].kernel_bottleneck.data.copy_(kernel_cur)

                # mean and std
                mean_cur = deepcopy(teacher.cv_nets[cv_n].whitening_layer.weights_mean)
                std_cur = deepcopy(teacher.cv_nets[cv_n].whitening_layer.weights_std)
                self.cv_nets[cv_n].whitening_layer.weights_mean.data.copy_(mean_cur)
                self.cv_nets[cv_n].whitening_layer.weights_std.data.copy_(std_cur)


class Mean_std_layer(torch.nn.Module):
    """Custom Linear layer for substracting the mean and dividing by
    the std

    Parameters
    ----------
    size_in: int
        The input size of which mean should be subtracted
    mean: torch.Tensor
        The mean value of the input training values
    std: torch.Tensor
        The std value of the input training values
    """

    def __init__(self, size_in, mean=None, std=None, mode_reverse=False):
        super().__init__()
        self.size_in = size_in
        if mean is None:
            mean = torch.zeros((1, size_in))
        self.weights_mean = torch.nn.Parameter(
            mean, requires_grad=False
        )  # nn.Parameter is a Tensor that's a module parameter.
        if std is None:
            std = torch.ones((1, size_in))
        self.weights_std = torch.nn.Parameter(std, requires_grad=False)
        self.mode_reverse = mode_reverse

    def forward(self, x):
        if self.mode_reverse:
            y = x * self.weights_std + self.weights_mean
        else:
            y = (x - self.weights_mean) / self.weights_std
        return y

    def set_both(self, mean, std):
        new_params = [mean, std]
        with torch.no_grad():
            for i, param in enumerate(self.parameters()):
                new_param = new_params[i]
                param.copy_(torch.Tensor(new_param))


class AttentionLayer(torch.nn.Module):
    def __init__(
        self,
        dim_cv,
        dim_xyz,
        acti_sm=torch.nn.Softmax(dim=-1),
        acti=torch.nn.Tanh,
        acti_v=torch.nn.Identity(),
    ):
        super().__init__()
        self.dim = dim_xyz
        self.q = torch.nn.Linear(dim_cv, dim_xyz)
        self.k = torch.nn.Linear(dim_xyz, dim_xyz)
        self.v = torch.nn.Linear(dim_xyz, dim_xyz)
        self.v2 = torch.nn.Linear(dim_xyz, dim_xyz)
        self.acti = acti
        self.acti_v = acti_v
        self.acti_sm = acti_sm

    def forward(self, cv_features, xyz):
        q = self.acti(self.q(cv_features))
        k = self.acti(self.k(xyz))
        v = self.acti_v(self.v(xyz))
        qk = q * k
        qk = self.acti_sm(qk)
        qkv = qk * v
        qkv = self.acti(self.v2(qkv))
        return qkv


class SelfAttention(torch.nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        acti_sm=torch.nn.Softmax(dim=-1),
        acti=torch.nn.Identity(),
        acti_v=torch.nn.Identity(),
    ):
        super().__init__()
        self.q = torch.nn.Linear(dim_in, dim_out)
        self.k = torch.nn.Linear(dim_in, dim_out)
        self.v = torch.nn.Linear(dim_in, dim_out)
        self.acti = acti
        self.acti_sm = acti_sm
        self.acti_v = acti_v

    def forward(self, x):
        q = self.acti(self.q(x))
        k = self.acti(self.k(x))
        v = self.acti_v(self.v(x))
        qk = q * k
        qk = self.acti_sm(qk)
        qkv = qk * v
        return qkv


class SelfAttentionRewire(torch.nn.Module):
    def __init__(
        self,
        dim_input,
        dim_input2,
        dim_interaction,
        acti_sm=torch.nn.Softmax(dim=-1),
        acti=torch.nn.Tanh(),
        acti_v=torch.nn.Identity(),
    ):
        super().__init__()
        self.q = torch.nn.Linear(dim_input2, dim_interaction)
        self.k = torch.nn.Linear(dim_input, dim_interaction)
        self.v = torch.nn.Linear(dim_input, dim_interaction)
        self.acti = acti
        self.acti_v = acti_v
        self.acti_sm = acti_sm

    def forward(self, input1, input2):
        q = self.acti(self.q(input2))
        k = self.acti(self.k(input1))
        v = self.acti_v(self.v(input1))
        qk = q * k
        qk = self.acti_sm(qk)
        qkv = qk * v
        return qkv
