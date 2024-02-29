"""
This module contains the core implementation of linear SFA.
"""
import warnings
import numpy as np
import math
import scipy as sp
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.utils.validation import check_array
from scipy.sparse import issparse
from sklearn.utils.multiclass import unique_labels
from sklearn.exceptions import NotFittedError
from sklearn.metrics import euclidean_distances


class SFA(TransformerMixin, BaseEstimator):
    """ Slow Feature Analysis (SFA)

    Linear dimensionality reduction and feature extraction method to be
    trained on time-series data. The data is decorrelated by whitening
    and linearly projected into the most slowly changing subspace.
    Slowness is measured by the average of squared one-step differences
    - thus, the most slowly changing subspacecorresponds to the directions
    with minimum variance for the dataset of one-step differences.
    It can be found using PCA.

    After training, the reduction can be applied to non-timeseries data as
    well. Read more in the :ref:`User Guide <SFA>`

    Parameters
    ----------
    n_components : int, float, None or str
        Number of components to keep.
        If n_components is not set all components are kept.

    batch_size : int or None
        Batches of the provided sequence that should be considered individual
        time-series. If batch_size is not set, the whole sequence will be
        considered connected.

    copy : bool, default=True
        If False, data passed to fit are overwritten and running
        fit(X).transform(X) will not yield the expected results,
        use fit_transform(X) instead.

    svd_solver : str {'auto', 'full', 'arpack', 'randomized'}
        The solver used by the internal PCA transformers.
        If auto :
            The solver is selected by a default policy based on `X.shape` and
            `n_components`: if the input data is larger than 500x500 and the
            number of components to extract is lower than 80% of the smallest
            dimension of the data, then the more efficient 'randomized'
            method is enabled. Otherwise the exact full SVD is computed and
            optionally truncated afterwards.
        If full :
            run exact full SVD calling the standard LAPACK solver via
            `scipy.linalg.svd` and select the components by postprocessing
        If arpack :
            run SVD truncated to n_components calling ARPACK solver via
            `scipy.sparse.linalg.svds`. It requires strictly
            0 < n_components < min(X.shape)
        If randomized :
            run randomized SVD by the method of Halko et al.

    tol : float >= 0, optional (default .0)
        Tolerance for singular values computed by svd_solver == 'arpack'
        of the internal PCA transformers.

    iterated_power : int >= 0, or 'auto', (default 'auto')
        Number of iterations for the power method computed by
        svd_solver == 'randomized'.

    random_state : int, RandomState instance, default=None
        Used when ``svd_solver`` == 'arpack' or 'randomized'. Pass an int
        for reproducible results across multiple function calls.

    robustness_cutoff : float, default=1e-15
        Applied after whitening. Features that have less explained
        variance in the input signal than robustness_cutoff will be
        considered trivial features and will be replaced in a way
        specified by fill_mode to ensure consistent output-dimensionality.

    fill_mode : str {'zero', 'fastest', 'noise'} or\
            None (default 'noise')
        The signal by which to replace output components that are
        artificial dimensions with almost no explained variance in
        input signal.
        If zero :
            Output components will be replace with constant zero signals.
            Subsequent applications of SFA will pick this up as a trivial
            signal again.
        If fastest :
            Output components will be replaced with copies of the fastest
            signals. Subsequent applications of SFA will pick this up as
            linearly dependent input dimension Since it is faster than
            any other signal, it would not be extracted in any case.
        If noise :
            Output components will be replaced with independent streams of
            Gaussian noise. The streams are typically not heavily correlated,
            but are very fast and thus will not be extracted.
        If fill_mode is specifically set to None :
            An exception will be thrown should trivial features are present
            in the input data.


    Attributes
    ----------
    input_dim_ : int
        The number of input features.

    delta_values_ : array, shape (n_components,)
        The estimated delta values (mean squared time-difference) of
        the different components.

    n_nontrivial_components_ : int
        The number of components that did not fall under the threshold
        defined by robustness_cutoff. Read a: effective dimension of
            input data.


    Examples
    --------
    >>> from sksfa import SFA
    >>> import numpy as np
    >>>
    >>> t = np.linspace(0, 8*np.pi, 1000).reshape(1000, 1)
    >>> t = t * np.arange(1, 6)
    >>>
    >>> ordered_cosines = np.cos(t)
    >>> mixed_cosines = np.dot(ordered_cosines, np.random.normal(0, 1, (5, 5)))
    >>>
    >>> sfa = SFA(n_components=2)
    >>> unmixed_cosines = sfa.fit_transform(mixed_cosines)
    """
    def __init__(self, n_components=None, *, batch_size=None, copy=True,
                 svd_solver="full", tol=.0, iterated_power="auto",
                 random_state=None, robustness_cutoff=1e-15,
                 fill_mode="noise"):
        self.n_components = n_components
        self.svd_solver = svd_solver
        self.copy = copy
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state
        self.batch_size = batch_size
        self.robustness_cutoff = robustness_cutoff
        self.fill_mode = fill_mode

    def _initialise_pca(self):
        """ Initialises internally used PCA transformers.
        """
        # initialize internal pca methods
        self.pca_whiten_ = PCA(svd_solver=self.svd_solver,
                               tol=self.tol, whiten=True)
        # initialize internal pca methods
        if self.batch_size is None:
            self.pca_diff_ = PCA(svd_solver=self.svd_solver, tol=self.tol)
        else:
            self.pca_diff_ = IncrementalPCA()

    def fit(self, X, tau, y=None):
        """Fit the model to X

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            The training input samples. If batch_size was set on init,
            this is assumed to be composed of concatenated time-series
            of batch_size length.

        y : None or {array-like}, shape (n_samples, 1)
            This does nothing and is only in here for compliance with
            sklearn API.

        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X, dtype=[np.float64, np.float32], ensure_2d=True,
                        copy=self.copy, ensure_min_features=1)
        # check_estimators test expects feature warnings before sample warnings
        X = check_array(X, dtype=[np.float64, np.float32], ensure_2d=True,
                        copy=self.copy,
                        ensure_min_samples=10 if self.batch_size is None
                        else self.batch_size)
        self.input_dim_ = X.shape[1]
        if self.n_components is None:
            self.n_components_ = self.input_dim_
        else:
            self.n_components_ = self.n_components
        self._initialise_pca()
        if issparse(X):
            raise TypeError('SFA does not support sparse input.')
        #if (X.shape[1] < 2):
        #    raise ValueError(f"At least two dimensional data is needed, "\
        #            "n_features={X.shape[1]} is too small.")
        self._fit(X, tau)
        self.is_fitted_ = True
        return self

    def partial(self, X, y=None):
        """Fit the model on a part of your dataset. This is used mainly for
        processing in hierarchical SFA.
        This is not the same as 'fit' using a batch-size.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            The training input samples. If batch_size was set on init,
            this is assumed to be composed of concatenated time-series
            of batch_size length.
        """
        n_samples = X.shape[0]
        assert(not self.is_fitted)
        if not self.is_partially_fitted:
            self._diff_sum = None
            self._n_partial_samples = 0
            self._n_partial_diff_samples = 0
            self._partial_eigenvectors = None
            self._partial_eigenvalues = None
        if self.batch_size is None:
            self._accumulate(X)
            self.last_partial_sample = X[-1]
        else:
            n_batches = math.ceil(n_samples/self.batch_size)
            for batch_idx in range(n_batches):
                current_batch = X[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
                self._accumulate(current_batch)
                self.last_partial_sample = current_batch[-1]
        self.is_partially_fitted_ = True
        return self

    def _accumulate(self, X):
        """ This function is meant for mini-batch training and collects covariance
        and mean information of the provided mini-batches.
        Training is finished as soon as transform is called or if the model has been
        fitted by full-batch training.
        """
        diff_X = X[1:] - X[:-1]
        if self._n_partial_samples == 0:
            self.sample_sum = X.sum(axis=0)
            self.outer_sum = np.dot(X.T, X)
            self._n_partial_samples = X.shape[0]
        else:
            self.sample_sum += X.sum(axis=0)
            self.outer_sum += np.dot(X.T, X)
            self._n_partial_samples += X.shape[0]

        if self._n_partial_diff_samples == 0:
            self.sample_diff_sum = diff_X.sum(axis=0)
            self.outer_diff_sum = np.dot(diff_X.T, diff_X)
            self._n_partial_diff_samples = diff_X.shape[0]
        else:
            last_partial_diff = X[0] - self.last_partial_sample
            self.sample_diff_sum += diff_X.sum(axis=0)# + last_partial_diff
            self.outer_diff_sum += np.dot(diff_X.T, diff_X)# + np.dot(last_partial_diff[:, None], last_partial_diff[None, :])
            self._n_partial_diff_samples += diff_X.shape[0]# + 1
        return self

    def _fit(self, X, tau):
        """Fit the model to X either by using minor component extraction
        on the difference time-series of the whitened data or other
        (not yet implemented) method.
        This is not meant for mini-batch training.
        For mini-batch training, use partial_fit instead.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            The training input samples. If batch_size was set on init,
            this is assumed to be composed of concatenated time-series
            of batch_size length.
        """
        self._fit_standard_method(X, tau)

    def _fit_standard_method(self, X, tau):
        """ Fit the model to X by first whitening the data, calculating
        the one-step differences and then extract their minor components.
        This method is not used for mini-batch training.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            The training input samples. If batch_size was set on init,
            this is assumed to be composed of concatenated time-series
            of batch_size length.
        """
        n_samples, input_dim = X.shape
        self.pca_whiten_.fit(X)
        X_whitened = self.pca_whiten_.transform(X)

        # Find non-trivial components
        input_evr = self.pca_whiten_.explained_variance_ratio_
        nontrivial_indices = np.argwhere(input_evr > self.robustness_cutoff)
        self.nontrivial_indices_ = nontrivial_indices.reshape((-1,))
        self.n_nontrivial_components_ = self.nontrivial_indices_.shape[0]
        self.n_trivial_ = input_dim - self.n_nontrivial_components_
        X_whitened = X_whitened[:, self.nontrivial_indices_]

        X_diff = X_whitened[tau:] - X_whitened[:-tau]
        self._diff_mean = X_diff.mean(axis=0)
        X_diff -= self._diff_mean

        if self.batch_size is None:
            self.pca_diff_.fit(X_diff)
        else:
            n_batches = int(n_samples/self.batch_size)
            current_slowness = 0
            for batch_idx in range(n_batches):
                batch_start = batch_idx * self.batch_size
                batch_end = (batch_idx + 1) * self.batch_size
                current_batch = X_whitened[batch_start:batch_end]
                batch_diff = current_batch[1:] - current_batch[:-1]
                self.pca_diff_.partial_fit(batch_diff - self._diff_mean)
        if self.n_nontrivial_components_ == 0:
            raise ValueError(f"While whitening, only trivial components were \
                    found. This can be caused by passing 0-only input.")
        if self.n_nontrivial_components_ < self.n_components_:
            warning_string = f"During whitening, {self.n_trivial_} trivial components "\
                    "with roughly zero explained variance have been found. This "\
                    "probably means that the effective dimension of your input is "\
                    "too low to find the desired {self.n_components_} slow features. "\
                    "Ways to deal with this are:\n\tInjecting noise via the 'noise_std' "\
                    "parameter.\n\tProviding more data.\n\tLowering the 'n_components' "\
                    "parameter.\n\tLowering the threshold 'robustness_cutoff'."
            if self.fill_mode is not None:
                warning_string += f"\nSince 'fill_mode' is set to {self.fill_mode}, "\
                        "missing output features will be filled by "
                if self.fill_mode == "zero":
                    warning_string += "a 0 signal."
                if self.fill_mode == "fastest":
                    warning_string += "duplicates of the fastest signal."
                if self.fill_mode == "noise":
                    warning_string += "independent white-noise."
                warnings.warn(warning_string, RuntimeWarning)
            else:
                warning_string += "\n\tSetting 'fill_mode' to replace trivial components\
                        with uninformative signals."
                raise ValueError(warning_string)
        self._compute_delta_values()

    def _fit_generalized_eigenmethod(self):
        """ For mini-batch training, the standard method is not used.
        Instead, a generalized eigenvalue problem from accumulated covariance
        matrices is solved.
        """
        C = (self.outer_sum  - np.outer(self.sample_sum, self.sample_sum)/(self._n_partial_samples)) / (self._n_partial_samples - 1)
        Cdot = self.outer_diff_sum/(self._n_partial_diff_samples - 1)
        self._partial_eigenvalues, self._partial_eigenvectors = sp.linalg.eigh(Cdot, C)

    def _compute_delta_values(self):
        """ Computes the delta values, but in compliance with the method
            chosen for handling trivial components.
        """
        delta_values_ = self.pca_diff_.explained_variance_[::-1]
        n_missing_components = self.n_components_ - self.n_nontrivial_components_
        if n_missing_components > 0:
            if self.fill_mode == "zero":
                delta_values_ = np.pad(delta_values_,
                                       (0, n_missing_components), "constant",
                                       constant_values=np.nan)
            if self.fill_mode == "fastest":
                delta_values_ = np.pad(delta_values_,
                                       (0, n_missing_components),
                                       "constant",
                                       constant_values=delta_values_[-1])
            if self.fill_mode == "noise":
                delta_values_ = np.pad(delta_values_,
                                       (0, n_missing_components),
                                       "constant",
                                       constant_values=2)
        self.delta_values_ = delta_values_

    def transform(self, X):
        """ Use the trained model to apply dimensionality reduction to X.
        First, it is whitened using a trained PCA model. Afterwards, it
        is projected onto the previously extracted slow subspace using a
        second trained PCA model.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The samples to be transformed.

        Returns
        -------
        y : ndarray, shape (n_samples, n_components)
            The slow features extracted from X.
        """
        X = check_array(X, dtype=[np.float64, np.float32],
                        ensure_2d=True, copy=self.copy)
        if (self.is_fitted) and (not self.is_partially_fitted):
            y = self.pca_whiten_.transform(X)
            y = y[:, self.nontrivial_indices_]
            y = self.pca_diff_.transform(y)
            n_missing_components = max(self.n_components_ - y.shape[1], 0)
            if n_missing_components > 0:
                if self.fill_mode == "zero":
                    y = np.pad(y, ((0, 0), (n_missing_components, 0)))
                if self.fill_mode == "fastest":
                    y = np.pad(y, ((0, 0), (n_missing_components, 0)), "edge")
                if self.fill_mode == "noise":
                    missing = np.random.normal(0, 1, (y.shape[0], n_missing_components))
                    missing -= missing.mean(axis=0)
                    missing /= missing.std(axis=0)
                    y = np.hstack([missing, y])
                return y[:, ::-1]
            else:
                y = y[:, -self.n_components_:][:, ::-1]
        elif (self.is_partially_fitted):
            if not (self.is_fitted):
                # fit using accumulated covariance matrices
                self._fit_generalized_eigenmethod()
                self.is_fitted_ = True
            y = np.dot(X, self._partial_eigenvectors[:, :self.n_components]) - np.dot(self.sample_sum/(self._n_partial_samples), self._partial_eigenvectors[:, :self.n_components])
        else:
            raise NotFittedError("SFA has not yet been fitted via 'fit' or 'partial_fit'")
        return y

    @property
    def is_fitted(self):
        try:
            return self.is_fitted_
        except AttributeError:
            return False

    @property
    def is_partially_fitted(self):
        try:
            return self.is_partially_fitted_
        except AttributeError:
            return False

    def affine_parameters(self):
        """
        This function provides the parameters of a trained linear SFA model.
        Only works if the standard 'fit' method has been used (no 'partial_fit') and trivial dimensions are either not filled or only filled in 'fill_mode' == 'zero'.
        Returns the parameters W, b so that the "np.dot(data, W.T) + b" is equivalent to calling the 'transform' method.
        """
        if not self.is_fitted:
            raise AttributeError("Transformer has to be fitted to data before extracting parameters")
        out_dim = self.pca_diff_.components_.shape[0]
        n_missing_components = max(self.n_components_ - out_dim, 0)
        print(n_missing_components)
        if n_missing_components > 0 and self.fill_mode not in (None, "zero"):
            raise RuntimeError("Affine parameters cannot be extracted: missing output dimensions are filled with something else then a 0-signal. Please set 'fill_mode' to 'zero' to avoid this or make sure that your input data has high enough rank.")
        W_whiten = (self.pca_whiten_.components_.T / np.sqrt(self.pca_whiten_.explained_variance_)).T[self.nontrivial_indices_]
        b_whiten = np.dot(-self.pca_whiten_.mean_, W_whiten.T)
        W_diff = self.pca_diff_.components_[::-1][:self.n_components_]
        b_diff = np.dot(- self.pca_diff_.mean_, W_diff.T)[:self.n_components_]
        W_transposed = np.dot(W_whiten.T, W_diff.T)
        b = np.dot(b_whiten, W_diff.T) + b_diff
        if n_missing_components > 0 and self.fill_mode == "zero":
            W_transposed = np.hstack([W_transposed, np.zeros((W_transposed.shape[0], n_missing_components))])
            b = np.hstack([b, np.zeros((n_missing_components,))])
        return W_transposed.T, b
        

