"""Collection of transformers for time series data."""

import numpy as np
from scipy import special


def log_transform(x):
    """Apply log transform of input series."""
    return np.log(x)


def inverse_log_transform(x):
    """Apply inverse log transform of input series."""
    return np.exp(x)


def boxcox_transform(x, l):
    """Apply Box-Cox(l) transformation."""
    transformed, _ = special.boxcox(x, lmbda=l)
    return transformed


def inverse_boxcox_transform(x, l):
    """Apply inverse Box-Cox(l) transformation."""
    transformed = special.inv_boxcox(x, lmbda=l)
    return transformed


def difference(x, order=1, seasonal_lag=None):
    """Compute seasonal and non-seasonal differences."""
    diff = x.copy()
    if seasonal_lag is not None:
        diff = x[seasonal_lag:] - x[:-seasonal_lag]
    return np.diff(diff, n=order)
