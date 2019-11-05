"""
Module for seasonality decomposition.

The following measures for trend and seasonality are defined:
- F_T = max {0, 1 - Var(R_t)/(Var(T_t + R_t)) }
- F_S = max {0, 1 - Var(R_t)/(Var(S_t + R_t)) }

See "Rob Hyndman: Forecasting Principles and Practice" for references.

"""
from statsmodels.tsa.api import STL, seasonal_decompose


def decompose(x, method="STL", **kwargs):
    """Perform seasonal decomposition of the time series."""
    if method not in ["STL", "MA"]:
        raise ValueError("`method` must be either 'STL' or 'MA'.")
    decomposition = None
    if method == "STL":
        decomposition = STL(x, **kwargs).fit()
    else:
        decomposition = seasonal_decompose(x, **kwargs)
    return decomposition


def seasonality_strength(x, **kwargs):
    """Return seasonality strength in [0,1], as defined in Rob Hyndman book."""
    decomposition = STL(x, **kwargs).fit()
    residual_std = decomposition.resid.std()
    strength = residual_std / (decomposition.seasonal.std() + residual_std)
    return max(0, 1 - strength) ** 2


def trend_strength(x, **kwargs):
    """Return trend strength in [0,1], as defined in Rob Hyndman book."""
    decomposition = STL(x, **kwargs).fit()
    residual_std = decomposition.resid.std()
    strength = residual_std / (decomposition.trend.std() + residual_std)
    return max(0, 1 - strength) ** 2


if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # Demo of shared x-axis for acf-pacf subplots.
    x = np.random.rand(100)
    x = pd.DataFrame(data=x, index=pd.DatetimeIndex(
        start="1/1/1990", periods=100, freq="M"))
    decomp = decompose(x, period=5)
    fig = decomp.plot()
    axes = fig.axes
    for i in range(len(axes) - 1):
        axes[i].get_shared_x_axes().join(axes[i], axes[i + 1])
    plt.show()
