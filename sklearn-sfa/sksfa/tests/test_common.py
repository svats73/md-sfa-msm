import pytest

from sklearn.utils.estimator_checks import check_estimator

from sksfa import SFA
from sklearn.decomposition import PCA, IncrementalPCA


@pytest.mark.parametrize("Estimator", [SFA])
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
