"""Module used to find the optimal parameters for ARIMA models.

Methodology:
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

import argparse
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
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

parser = argparse.ArgumentParser()
parser.add_argument("dataset_index", type=int, choices=range(6))
parser.add_argument("--period", type=int)
parser.add_argument("--description", action="store_true")
parser.add_argument("--time_plots", action="store_true")
parser.add_argument("--lmbda", type=float)
parser.add_argument("--autocorr_tests", action="store_true")
parser.add_argument("--diff_tests", action="store_true")
parser.add_argument("--decomposition", action="store_true")
group = parser.add_mutually_exclusive_group()
group.add_argument("--fit", action="store_true")
group.add_argument("--auto_fit", action="store_true")
parser.add_argument('--order', nargs=3, type=int)
parser.add_argument('--seasonal_order', nargs=3, type=int)
parser.add_argument('--seasonal', action="store_true")
args = parser.parse_args()

if args.time_plots and args.lmbda is None:
    print("Please specify lambda value for Box-Cox transform.")
    sys.exit(0)

if args.order is not None:
    args.order = tuple(args.order)
if args.seasonal_order is not None:
    args.seasonal_order = tuple(args.seasonal_order) + (args.period,)

if args.auto_fit and args.period is None:
    print("Please specify period.")
    sys.exit(0)
if args.fit and args.order is None:
    print("Please specify order.")
    sys.exit(0)

dataset_names = ["iio_us-east-1_i-a2eb1cd9_NetworkIn.csv",
                 "machine_temperature_system_failure.csv",
                 "ec2_cpu_utilization_fe7f93.csv",
                 "rds_cpu_utilization_e47b3b.csv",
                 "grok_asg_anomaly.csv",
                 "elb_request_count_8c0756.csv"]

dataset_name = dataset_names[args.dataset_index]
print("Chosen dataset:", dataset_name)

reader = NABReader()
reader.load_data()
reader.load_labels()

# Load data and labels
df = reader.data.get(dataset_name)
labels = reader.labels.get(dataset_name)
labels_windows = reader.label_windows.get(dataset_name)

print(df.head())

# Train-test split
train_percentage = 0.8
train_len = int(train_percentage * len(df))
train, test = df.value[:train_len], df.value[train_len:]

# Describe data
if args.description:
    print("Dataframe information:")
    df.info()
    print("Data description:")
    print(df.describe())

# Time plots
if args.time_plots:
    time_plot(df.value, scale=True)
    log_value = log_transform(df.value)
    time_plot(log_value, scale=True, label="log(value)")
    lmbda = args.lmbda
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
if args.autocorr_tests:
    lags = 50
    fig, axes = plt.subplots(2, 1, sharex=True)
    plot_acf(df.value, lags=lags, ax=axes[0])
    plot_pacf(df.value, lags=lags, ax=axes[1])
    box_ljung(df.value, nlags=10).format()

# ADF-KPSS unit root tests / differencing
if args.diff_tests:
    diff = difference(df.value, order=1, seasonal_lag=None)
    fig, axes = plt.subplots(2, 1, sharex=True)
    time_plot(df.value, ax=axes[0], label="original")
    time_plot(diff, ax=axes[1], label="differenced")
    for ax in axes:
        ax.legend()
    plt.gcf().suptitle("Original data vs differenced data")
    adf(diff).format()
    kpss(diff).format()

# Seasonality decomposition
if args.decomposition:
    # 60	1440	10080
    freq = df.index.inferred_freq
    print("Inferred frequency:", freq)
    # period = int(60 / 5)
    print(
        "Seasonality strength:",
        seasonality_strength(df.value, period=args.period))
    print("Trend strength:",
          trend_strength(df.value, period=args.period))

    ma_decomposition = decompose(df.value, method="MA", period=args.period)
    stl_decomposition = decompose(df.value, period=args.period)
    robust_stl_decomposition = decompose(
        df.value, robust=True, period=args.period)
    stl_decomposition.plot()
    plt.gcf().suptitle('Non-robust STL seasonality decomposition')
    ma_decomposition.plot()
    plt.gcf().suptitle('Moving average seasonality decomposition')
    robust_stl_decomposition.plot()
    plt.gcf().suptitle('Robust STL seasonality decomposition')

# Fit ARIMA model
if args.fit or args.auto_fit:
    if args.auto_fit:
        arima = auto_arima(
            train,
            stepwise=True,
            trace=1,
            m=args.period,
            information_criterion="aicc",
            seasonal=args.seasonal,
            error_action="ignore",
            suppress_warnings=True,
        )
        print(arima.summary())
    elif args.fit:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arima = ARIMA(order=args.order,
                          seasonal_order=args.seasonal_order)
            arima.fit(train)

        residuals = arima.resid()
        print("train lengths: data={} resid={}".format(
            train_len, residuals.shape[0]))
        len_delta = train_len - residuals.shape[0]

    # Diagnostics plot
    arima.plot_diagnostics(lags=50)
    box_ljung(residuals, nlags=20).format()
    plt.gcf().suptitle('Diagnostics Plot')
    plt.figure()
    plt.plot(df.value.index[len_delta:train_len],
             np.abs(residuals), label="abs(residuals)")
    plt.plot(df.value, label="data", alpha=0.5)
    # fig, axes = plt.subplots(3, 1, sharex=True)
    # axes[0].plot(arima.resid(), label="residuals")
    # axes[1].plot(arima.resid()**2, label="residuals^2")
    # axes[2].plot(np.abs(arima.resid()), label="abs(residuals)")
    # for ax in axes:
    #     ax.legend()
    plt.legend()
    plt.gcf().suptitle('Residuals Plot')

    # Plot fitted values and forecasts
    predictions = arima.predict(n_periods=test.shape[0])
    fitted_values = arima.predict_in_sample()
    print("train lengths: data={} fitted_values={}".format(
        train_len, fitted_values.shape[0]))
    len_delta = train_len - fitted_values.shape[0]

    plt.figure()
    plt.plot(df.value.index[train_len:], test,
             '--', color='C0', label="test set")
    plt.plot(df.value.index[train_len:], predictions,
             '--', color='C1', label="forecasted values")
    plt.plot(df.value.index[:train_len], train, color='C0', label="train set")
    plt.plot(df.value.index[len_delta:train_len], fitted_values,
             color='C1', label="fitted values")
    plt.legend()
    plt.gcf().suptitle("Fitted values and forecasts")

plt.show()
