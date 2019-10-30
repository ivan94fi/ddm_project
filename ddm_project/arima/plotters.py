"""Collection of functions to plot time series data."""

import matplotlib.pyplot as plt
import statsmodels.api as sm


def plot_acf(series, ax=None, lags=None, marker=None):
    """Plot the autocorrelation funciton."""
    fig = sm.graphics.tsa.plot_acf(series, ax=ax, lags=lags, marker=marker)
    return fig


def plot_pacf(series, ax=None, lags=None, marker=None):
    """Plot the partial autocorrelation funciton."""
    fig = sm.graphics.tsa.plot_pacf(series, ax=ax, lags=lags, marker=marker)
    return fig


def time_plot(series, ax=None, label=None, scale=False, **kwargs):
    """Plot the values of the series."""
    if ax is None:
        ax = plt.gca()
    if label is None:
        try:
            label = series.name
        except AttributeError:
            pass
    if scale:
        scaled_series = (series - series.mean()) / series.std()
    else:
        scaled_series = series
    ax.plot(scaled_series, label=label, **kwargs)
    return ax
