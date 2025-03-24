import torch
import numpy as np

from typing import Optional, Union, Callable, Tuple


eigh = torch.linalg.eigh if hasattr(torch, 'linalg') else lambda x: torch.symeig(x, eigenvectors=True)
multi_dot = torch.linalg.multi_dot if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'multi_dot') else \
    lambda args: torch.chain_matmul(*args)

def symeig_reg(mat, epsilon: float = 1e-6, mode='regularize', eigenvectors=True) \
        -> Tuple["torch.Tensor", Optional["torch.Tensor"]]:
    r""" Solves a eigenvector/eigenvalue decomposition for a hermetian matrix also if it is rank deficient.

    Parameters
    ----------
    mat : torch.Tensor
        the hermetian matrix
    epsilon : float, default=1e-6
        Cutoff for eigenvalues.
    mode : str, default='regularize'
        Whether to truncate eigenvalues if they are too small or to regularize them by taking the absolute value
        and adding a small positive constant. :code:`trunc` leads to truncation, :code:`regularize` leads to epsilon
        being added to the eigenvalues after taking the absolute value
    eigenvectors : bool, default=True
        Whether to compute eigenvectors.

    Returns
    -------
    (eigval, eigvec) : Tuple[torch.Tensor, Optional[torch.Tensor]]
        Eigenvalues and -vectors.
    """
    assert mode in sym_inverse.valid_modes, f"Invalid mode {mode}, supported are {sym_inverse.valid_modes}"

    if mode == 'regularize':
        identity = torch.eye(mat.shape[0], dtype=mat.dtype, device=mat.device)
        mat = mat + epsilon * identity

    # Calculate eigvalues and potentially eigvectors
    eigval, eigvec = eigh(mat)

    if eigenvectors:
        eigvec = eigvec.transpose(0, 1)

    if mode == 'trunc':
        # Filter out Eigenvalues below threshold and corresponding Eigenvectors
        mask = eigval > epsilon
        eigval = eigval[mask]
        if eigenvectors:
            eigvec = eigvec[mask]
    elif mode == 'regularize':
        # Calculate eigvalues and eigvectors
        eigval = torch.abs(eigval)
    elif mode == 'clamp':
        eigval = torch.clamp_min(eigval, min=epsilon)

    return eigval, eigvec


def sym_inverse(mat, epsilon: float = 1e-6, return_sqrt=False, mode='regularize'):
    """ Utility function that returns the inverse of a matrix, with the
    option to return the square root of the inverse matrix.

    Parameters
    ----------
    mat: numpy array with shape [m,m]
        Matrix to be inverted.
    epsilon : float
        Cutoff for eigenvalues.
    return_sqrt: bool, optional, default = False
        if True, the square root of the inverse matrix is returned instead
    mode: str, default='trunc'
        Whether to truncate eigenvalues if they are too small or to regularize them by taking the absolute value
        and adding a small positive constant. :code:`trunc` leads to truncation, :code:`regularize` leads to epsilon
        being added to the eigenvalues after taking the absolute value

    Returns
    -------
    x_inv: numpy array with shape [m,m]
        inverse of the original matrix
    """
    eigval, eigvec = symeig_reg(mat, epsilon, mode)

    # Build the diagonal matrix with the filtered eigenvalues or square
    # root of the filtered eigenvalues according to the parameter
    if return_sqrt:
        diag = torch.diag(torch.sqrt(1. / eigval))
    else:
        diag = torch.diag(1. / eigval)

    return multi_dot([eigvec.t(), diag, eigvec])


sym_inverse.valid_modes = ('trunc', 'regularize', 'clamp')

def koopman_matrix(x: "torch.Tensor", y: "torch.Tensor", weights: "torch.Tensor" = None, sym=False, epsilon: float = 1e-6, mode: str = 'trunc',
                   c_xx: Optional[Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]] = None) -> "torch.Tensor":
    r""" Computes the Koopman matrix

    .. math:: K = C_{00}^{-1/2}C_{0t}C_{tt}^{-1/2}

    based on data over which the covariance matrices :math:`C_{\cdot\cdot}` are computed.

    Parameters
    ----------
    x : torch.Tensor
        Instantaneous data.
    y : torch.Tensor
        Time-lagged data.
    epsilon : float, default=1e-6
        Cutoff parameter for small eigenvalues.
    mode : str, default='trunc'
        Regularization mode for Hermetian inverse. See :meth:`sym_inverse`.
    c_xx : tuple of torch.Tensor, optional, default=None
        Tuple containing c00, c0t, ctt if already computed.

    Returns
    -------
    K : torch.Tensor
        The Koopman matrix.
    """
    if c_xx is not None:
        c00, c0t, ctt = c_xx
    else:
        c00, c0t, ctt = covariances(x, y, remove_mean=True, weights=weights)
    if sym:
        c00 = 1/2 * (c00 + ctt)
        ctt = c00
        c0t = 1/2 * (c0t + c0t.t())
    c00_sqrt_inv = sym_inverse(c00, return_sqrt=True, epsilon=epsilon, mode=mode)
    ctt_sqrt_inv = sym_inverse(ctt, return_sqrt=True, epsilon=epsilon, mode=mode)
    return multi_dot([c00_sqrt_inv, c0t, ctt_sqrt_inv]).t()


def covariances(x: "torch.Tensor", y: "torch.Tensor", remove_mean: bool = True, weights=None):
    """Computes instantaneous and time-lagged covariances matrices.

    Parameters
    ----------
    x : (T, n) torch.Tensor
        Instantaneous data.
    y : (T, n) torch.Tensor
        Time-lagged data.
    remove_mean: bool, default=True
        Whether to remove the mean of x and y.

    Returns
    -------
    cov_00 : (n, n) torch.Tensor
        Auto-covariance matrix of x.
    cov_0t : (n, n) torch.Tensor
        Cross-covariance matrix of x and y.
    cov_tt : (n, n) torch.Tensor
        Auto-covariance matrix of y.

    See Also
    --------
    deeptime.covariance.Covariance : Estimator yielding these kind of covariance matrices based on raw numpy arrays
                                     using an online estimation procedure.
    """
    assert x.shape == y.shape, "x and y must be of same shape"
    batch_size = x.shape[0]
    if weights is None:
        weights = 1
    else:
        weights = weights / weights.sum()
        weights = torch.sqrt(weights)

    if remove_mean:
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)
    
    x *= weights
    y *= weights

    # Calculate the cross-covariance
    y_t = y.transpose(0, 1)
    x_t = x.transpose(0, 1)
    cov_01 = 1 / (batch_size - 1) * torch.matmul(x_t, y)
    # Calculate the auto-correlations
    cov_00 = 1 / (batch_size - 1) * torch.matmul(x_t, x)
    cov_11 = 1 / (batch_size - 1) * torch.matmul(y_t, y)

    return cov_00, cov_01, cov_11


valid_score_methods = ('VAMP1', 'VAMP2', 'VAMPE')


def vamp_score(data: "torch.Tensor", data_lagged: "torch.Tensor", weights: "torch.Tensor" = None, sym=False, method='VAMP2', epsilon: float = 1e-6, mode='trunc'):
    r"""Computes the VAMP score based on data and corresponding time-shifted data.

    Parameters
    ----------
    data : torch.Tensor
        (N, d)-dimensional torch tensor
    data_lagged : torch.Tensor
        (N, k)-dimensional torch tensor
    method : str, default='VAMP2'
        The scoring method. See :meth:`score <deeptime.decomposition.CovarianceKoopmanModel.score>` for details.
    epsilon : float, default=1e-6
        Cutoff parameter for small eigenvalues, alternatively regularization parameter.
    mode : str, default='trunc'
        Regularization mode for Hermetian inverse. See :meth:`sym_inverse`.

    Returns
    -------
    score : torch.Tensor
        The score. It contains a contribution of :math:`+1` for the constant singular function since the
        internally estimated Koopman operator is defined on a decorrelated basis set.
    """
    assert method in valid_score_methods, f"Invalid method '{method}', supported are {valid_score_methods}"
    assert data.shape == data_lagged.shape, f"Data and data_lagged must be of same shape but were {data.shape} " \
                                            f"and {data_lagged.shape}."
    out = None
    if method == 'VAMP1':
        koopman = koopman_matrix(data, data_lagged, weights=weights, sym=sym, epsilon=epsilon, mode=mode)
        out = torch.norm(koopman, p='nuc')
    elif method == 'VAMP2':
        koopman = koopman_matrix(data, data_lagged, weights=weights, sym=sym, epsilon=epsilon, mode=mode)
        out = torch.pow(torch.norm(koopman, p='fro'), 2)
    elif method == 'VAMPE':
        c00, c0t, ctt = covariances(data, data_lagged, remove_mean=True)
        if sym:
            c00 = 1/2 * (c00 + ctt)
            ctt = c00
            c0t = 1/2 * (c0t + c0t.t())
        c00_sqrt_inv = sym_inverse(c00, epsilon=epsilon, return_sqrt=True, mode=mode)
        ctt_sqrt_inv = sym_inverse(ctt, epsilon=epsilon, return_sqrt=True, mode=mode)
        koopman = multi_dot([c00_sqrt_inv, c0t, ctt_sqrt_inv]).t()

        u, s, v = torch.svd(koopman)
        mask = s > epsilon

        u = torch.mm(c00_sqrt_inv, u[:, mask])
        v = torch.mm(ctt_sqrt_inv, v[:, mask])
        s = s[mask]

        u_t = u.t()
        v_t = v.t()
        s = torch.diag(s)

        out = torch.trace(
            2. * multi_dot([s, u_t, c0t, v]) - multi_dot([s, u_t, c00, u, s, v_t, ctt, v])
        )
    assert out is not None
    return out


def vampnet_loss(data: "torch.Tensor", data_lagged: "torch.Tensor", weights: "torch.Tensor" = None, sym=False, method='VAMP2', epsilon: float = 1e-6,
                 mode: str = 'trunc'):
    r"""Loss function that can be used to train VAMPNets. It evaluates as :math:`-\mathrm{score}`. The score
    is implemented in :meth:`score`."""
    return -1. * vamp_score(data, data_lagged, weights=weights, sym=sym, method=method, epsilon=epsilon, mode=mode)


def symeig_reg_np(mat, epsilon: float = 1e-6, mode='regularize', eigenvectors=True) \
        -> Tuple["np.ndarray", Optional["np.ndarray"]]:
    r""" Solves a eigenvector/eigenvalue decomposition for a hermetian matrix also if it is rank deficient.

    Parameters
    ----------
    mat : torch.Tensor
        the hermetian matrix
    epsilon : float, default=1e-6
        Cutoff for eigenvalues.
    mode : str, default='regularize'
        Whether to truncate eigenvalues if they are too small or to regularize them by taking the absolute value
        and adding a small positive constant. :code:`trunc` leads to truncation, :code:`regularize` leads to epsilon
        being added to the eigenvalues after taking the absolute value
    eigenvectors : bool, default=True
        Whether to compute eigenvectors.

    Returns
    -------
    (eigval, eigvec) : Tuple[torch.Tensor, Optional[torch.Tensor]]
        Eigenvalues and -vectors.
    """
    assert mode in sym_inverse.valid_modes, f"Invalid mode {mode}, supported are {sym_inverse.valid_modes}"

    if mode == 'regularize':
        identity = np.eye(mat.shape[0], dtype=mat.dtype)
        mat = mat + epsilon * identity

    # Calculate eigvalues and potentially eigvectors
    eigval, eigvec = np.linalg.eigh(mat)

    if eigenvectors:
        eigvec = np.transpose(eigvec, (1,0))

    if mode == 'trunc':
        # Filter out Eigenvalues below threshold and corresponding Eigenvectors
        mask = eigval > epsilon
        eigval = eigval[mask]
        if eigenvectors:
            eigvec = eigvec[mask]
    elif mode == 'regularize':
        # Calculate eigvalues and eigvectors
        eigval = np.abs(eigval)
    elif mode == 'clamp':
        eigval = np.clip(eigval, a_min=epsilon, a_max=None)

    return eigval, eigvec


def sym_inverse_np(mat, epsilon: float = 1e-6, return_sqrt=False, mode='regularize'):
    """ Utility function that returns the inverse of a matrix, with the
    option to return the square root of the inverse matrix.

    Parameters
    ----------
    mat: numpy array with shape [m,m]
        Matrix to be inverted.
    epsilon : float
        Cutoff for eigenvalues.
    return_sqrt: bool, optional, default = False
        if True, the square root of the inverse matrix is returned instead
    mode: str, default='trunc'
        Whether to truncate eigenvalues if they are too small or to regularize them by taking the absolute value
        and adding a small positive constant. :code:`trunc` leads to truncation, :code:`regularize` leads to epsilon
        being added to the eigenvalues after taking the absolute value

    Returns
    -------
    x_inv: numpy array with shape [m,m]
        inverse of the original matrix
    """
    eigval, eigvec = symeig_reg_np(mat, epsilon, mode)

    # Build the diagonal matrix with the filtered eigenvalues or square
    # root of the filtered eigenvalues according to the parameter
    if return_sqrt:
        diag = np.diag(np.sqrt(1. / eigval))
    else:
        diag = np.diag(1. / eigval)

    return eigvec.T @ diag @ eigvec

def covariances_np(x: "np.ndarray", y: "np.ndarray", remove_mean: bool = True, weights=None):
    """Computes instantaneous and time-lagged covariances matrices.

    Parameters
    ----------
    x : (T, n) torch.Tensor
        Instantaneous data.
    y : (T, n) torch.Tensor
        Time-lagged data.
    remove_mean: bool, default=True
        Whether to remove the mean of x and y.

    Returns
    -------
    cov_00 : (n, n) torch.Tensor
        Auto-covariance matrix of x.
    cov_0t : (n, n) torch.Tensor
        Cross-covariance matrix of x and y.
    cov_tt : (n, n) torch.Tensor
        Auto-covariance matrix of y.

    See Also
    --------
    deeptime.covariance.Covariance : Estimator yielding these kind of covariance matrices based on raw numpy arrays
                                     using an online estimation procedure.
    """
    assert x.shape == y.shape, "x and y must be of same shape"
    batch_size = x.shape[0]
    if weights is None:
        weights = 1
    else:
        weights = weights / weights.sum()
        weights = np.sqrt(weights)

    if remove_mean:
        x = x - x.mean(axis=0, keepdims=True)
        y = y - y.mean(axis=0, keepdims=True)
    
    x *= weights
    y *= weights

    # Calculate the cross-covariance
    y_t = y.T
    x_t = x.T
    cov_01 = 1 / (batch_size - 1) * x_t @ y
    # Calculate the auto-correlations
    cov_00 = 1 / (batch_size - 1) * x_t @ x
    cov_11 = 1 / (batch_size - 1) * y_t @ y

    return cov_00, cov_01, cov_11

def transition_matrix(x: "np.ndarray", y: "np.ndarray", weights: "np.ndarray" = None, sym=False, epsilon: float = 1e-6, mode: str = 'trunc')-> Tuple["torch.Tensor", "torch.Tensor"]:

    x_mean = x.mean(0, keepdims=True)
    c00, c0t, ctt = covariances_np(x, y, remove_mean=True, weights=weights)
    if sym:
        c00 = 1/2 * (c00 + ctt)
        c0t = 1/2 * (c0t + c0t.T)
    c00_inv = sym_inverse_np(c00, return_sqrt=False, epsilon=epsilon, mode=mode)
    T = c00_inv @ c0t
    return T, x_mean