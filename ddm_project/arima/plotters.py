"""Collection of functions to plot time series data."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


def plot_acf(series, ax=None, lags=None, marker=None):
    """Plot the autocorrelation funciton."""
    fig = sm.graphics.tsa.plot_acf(series, ax=ax, lags=lags, marker=marker)
    return fig


def plot_pacf(series, ax=None, lags=None, marker=None):
    """Plot the partial autocorrelation funciton."""
    fig = sm.graphics.tsa.plot_acf(series, ax=ax, lags=lags, marker=marker)
    return fig


def time_plot(series, ax=None):
    """Plot the values of the series."""
    if ax is None:
        ax = plt.gca()
    label = None
    if series.name is not None:
        label = series.name
    ax.plot(series, label=label)
    return ax
