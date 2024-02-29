import pytest
import numpy as np

from sklearn.datasets import load_iris
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_allclose

from sksfa import SFA
import numpy as np


MAX_TEST_DIM = 10
MAX_TEST_N_KILOSAMPLES = 30

@pytest.fixture
def data():
    return mixed_trigonometric_functions()

def mixed_trigonometric_functions(dimension=5, n_samples=1000, rank_deficit=0):
    t = np.linspace(0, 2 * np.pi, n_samples).reshape(n_samples, 1)
    t = t * np.arange(1, dimension + 1)
    deficit_dimensions = np.random.choice(np.arange(0, dimension), rank_deficit, replace=False)
    t[:, deficit_dimensions] = t[:, deficit_dimensions] * 0
    trig_functions = np.cos(t)
    mixed_functions = np.dot(trig_functions, np.random.normal(0, 1, (dimension, dimension)))
    return mixed_functions

def compute_delta(data):
    X = np.copy(data)
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    return (X[1:] - X[:-1]).var(axis=0)

def test_sfa_is_fitted(data):
    est = SFA()
    est.fit(data)
    assert hasattr(est, 'is_fitted_')

@pytest.mark.parametrize("dimension,n_samples", [(dimension, 1000 * n_kilosamples) \
        for dimension in range(2, MAX_TEST_DIM) for n_kilosamples in range(1, MAX_TEST_N_KILOSAMPLES + 1, 5)])
def test_sfa_constraints(dimension, n_samples):
    current_data = mixed_trigonometric_functions(dimension, n_samples)
    sfa = SFA()
    slow_features = sfa.fit_transform(current_data)
    covariance_matrix = np.cov(slow_features.T)
    assert np.allclose(covariance_matrix, np.eye(dimension))

@pytest.mark.parametrize("dimension,n_samples", [(dimension, 1000 * n_kilosamples) \
        for dimension in range(2, MAX_TEST_DIM) for n_kilosamples in range(1, MAX_TEST_N_KILOSAMPLES + 1, 5)])
def test_sfa_delta_values(dimension, n_samples):
    current_data = mixed_trigonometric_functions(dimension, n_samples)
    sfa = SFA()
    slow_features = sfa.fit_transform(current_data)
    explicit_delta_values = compute_delta(slow_features)
    assert np.allclose(explicit_delta_values, sfa.delta_values_)

@pytest.mark.parametrize("dimension,n_samples", [(dimension, 1000 * n_kilosamples) \
        for dimension in range(2, MAX_TEST_DIM) for n_kilosamples in range(1, MAX_TEST_N_KILOSAMPLES + 1, 5)])
def test_sfa_feature_order(dimension, n_samples):
    current_data = mixed_trigonometric_functions(dimension, n_samples)
    sfa = SFA()
    slow_features = sfa.fit_transform(current_data)
    explicit_delta_values = compute_delta(slow_features)
    assert np.allclose(explicit_delta_values, np.sort(explicit_delta_values))

@pytest.mark.parametrize("dimension,rank_deficit", [(dimension, rank_deficit) \
        for dimension in range(2, MAX_TEST_DIM) for rank_deficit in range(1, dimension + 1)])
def test_sfa_detects_rank_deficit(dimension, rank_deficit):
    sfa = SFA(fill_mode=None)
    current_data = mixed_trigonometric_functions(dimension, rank_deficit=rank_deficit)
    with pytest.raises(ValueError):
        sfa.fit(current_data)

@pytest.mark.parametrize("dimension,n_samples", [(dimension, 1000 * n_kilosamples) \
        for dimension in range(2, MAX_TEST_DIM) for n_kilosamples in range(1, MAX_TEST_N_KILOSAMPLES + 1, 5)])
def test_sfa_parameter_computation(dimension, n_samples):
    current_data = mixed_trigonometric_functions(dimension, n_samples)
    sfa = SFA()
    slow_features = sfa.fit_transform(current_data)
    W, b = sfa.affine_parameters()
    affine_transformed = np.dot(current_data, W.T) + b
    assert np.allclose(slow_features, affine_transformed)

@pytest.mark.parametrize("dimension,rank_deficit", [(dimension, rank_deficit) \
        for dimension in range(2, MAX_TEST_DIM) for rank_deficit in range(1, dimension)])
def test_sfa_parameter_computation_rank_deficit_zero_fill(dimension, rank_deficit):
    current_data = mixed_trigonometric_functions(dimension, rank_deficit=rank_deficit)
    sfa = SFA(fill_mode="zero")
    slow_features = sfa.fit_transform(current_data)
    W, b = sfa.affine_parameters()
    affine_transformed = np.dot(current_data, W.T) + b
    assert np.allclose(slow_features, affine_transformed)

@pytest.mark.parametrize("dimension,rank_deficit", [(dimension, rank_deficit) \
        for dimension in range(2, MAX_TEST_DIM) for rank_deficit in range(1, dimension + 1)])
def test_sfa_parameter_computation_rank_deficit_nonzero_fill(dimension, rank_deficit):
    current_data = mixed_trigonometric_functions(dimension, rank_deficit=rank_deficit)
    sfa = SFA(fill_mode="noise")
    slow_features = sfa.fit_transform(current_data)
    with pytest.raises(RuntimeError):
      W, b = sfa.affine_parameters()
