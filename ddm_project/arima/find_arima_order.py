"""Module used to find the optimal parameters for ARIMA models.

Metodo:
- load
- describe
- time plot
- box-cox
- acf/pacf/box-ljung
- differencing/kpss/adfuller
- seasonality decomposition

- fit

- checkresiduals: AIC_c/ljung-box/ACF/uncorrelated/zero mean
"""

import matplotlib.pyplot as plt
import statsmodels.tsa.api as tsa
from pandas.plotting import register_matplotlib_converters
from pmdarima.arima import ARIMA, auto_arima

from ddm_project.arima.decomposition import (
    decompose, seasonality_strength, trend_strength
)
from ddm_project.arima.plotters import plot_acf, plot_pacf, time_plot
from ddm_project.arima.statistical_tests import adf, box_ljung, kpss
from ddm_project.arima.transformers import (
    boxcox_transform, difference, log_transform
)
from ddm_project.readers.nab_dataset_reader import NABReader

register_matplotlib_converters()

#####################
do_description = False
do_plots = False
do_acf_pacf_box_ljung = False
do_unit_root_tests_diff = False
do_seas_decomposition = True
#####################

reader = NABReader()
reader.load_data()
reader.load_labels()

# Load data
df = reader.data.get("iio_us-east-1_i-a2eb1cd9_NetworkIn.csv")

# Describe data
if do_description:
    print("Dataframe information:")
    df.info()
    print("Data description:")
    print(df.describe())

# Time plots
if do_plots:
    time_plot(df.value, scale=True)
    log_value = log_transform(df.value)
    time_plot(log_value, scale=True, label="log(value)")
    lmbda = 0.4
    boxcox_value = boxcox_transform(df.value, lmbda)
    time_plot(boxcox_value, scale=True,
              label="boxcox({})(value)".format(lmbda))
    plt.legend()
    plt.figure()
    time_plot(df.value)
    plt.legend()
    plt.figure()
    time_plot(log_value, label="log(value)")
    plt.legend()
    plt.figure()
    time_plot(boxcox_value, label="boxcox({})(value)".format(lmbda))
    plt.legend()


# ACF/PACF/Box-Ljung test
if do_acf_pacf_box_ljung:
    lags = 50
    fig, axes = plt.subplots(2, 1, sharex=True)
    plot_acf(df.value, lags=lags, ax=axes[0])
    plot_pacf(df.value, lags=lags, ax=axes[1])
    box_ljung(df.value, nlags=10).format()

# ADF-KPSS unit root tests / differencing
if do_unit_root_tests_diff:
    diff = difference(df.value, order=1, seasonal_lag=None)
    fig, axes = plt.subplots(2, 1, sharex=True)
    time_plot(df.value, ax=axes[0], label="original")
    time_plot(diff, ax=axes[1], label="differenced")
    plt.legend()
    adf(diff).format()
    kpss(diff).format()

# seasonality decomposition
if do_seas_decomposition:
    # 60	1440	10080
    freq = df.index.inferred_freq
    print("Inferred frequency:", freq)
    period = int(60 / 5)
    print("Seasonality strength:", seasonality_strength(df.value, period=period))
    print("Trend strength:", trend_strength(df.value, period=period))

    ma_decomposition = decompose(df.value, method="MA", period=period)
    stl_decomposition = decompose(df.value, period=period)
    robust_stl_decomposition = decompose(df.value, robust=True, period=period)
    stl_decomposition.plot()
    plt.gcf().suptitle('Non-robust STL seasonality decomposition')
    ma_decomposition.plot()
    plt.gcf().suptitle('Moving average seasonality decomposition')
    robust_stl_decomposition.plot()
    plt.gcf().suptitle('Robust STL seasonality decomposition')


plt.show()
