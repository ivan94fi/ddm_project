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
import collections
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
from ddm_project.metrics.metrics import get_nab_score, get_simple_metrics
from ddm_project.readers.nab_dataset_reader import NABReader
from ddm_project.utils.utils import _make_plots, get_gt_arrays

register_matplotlib_converters()

parser = argparse.ArgumentParser()
parser.add_argument("dataset_index", type=int, choices=range(6))
args = parser.parse_args()

dataset_names = ["iio_us-east-1_i-a2eb1cd9_NetworkIn.csv",
                 "machine_temperature_system_failure.csv",
                 "ec2_cpu_utilization_fe7f93.csv",
                 "rds_cpu_utilization_e47b3b.csv",
                 "grok_asg_anomaly.csv",
                 "elb_request_count_8c0756.csv"]

dataset_name = dataset_names[args.dataset_index]
print("Chosen dataset:", dataset_name)

#####################
do_description = False
do_plots = False
do_acf_pacf_box_ljung = False
do_unit_root_tests_diff = False
do_seas_decomposition = False
do_fit = False
do_evaluate = True
#####################

reader = NABReader()
reader.load_data()
reader.load_labels()

# Load data and labels
df = reader.data.get(dataset_name)
labels = reader.labels.get(dataset_name)
labels_windows = reader.label_windows.get(dataset_name)

# Train-test split
train_percentage = 0.8
train_len = int(train_percentage * len(df))
train, test = df.value[:train_len], df.value[train_len:]

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

# Seasonality decomposition
if do_seas_decomposition:
    # 60	1440	10080
    freq = df.index.inferred_freq
    print("Inferred frequency:", freq)
    period = int(60 / 5)
    print(
        "Seasonality strength:", seasonality_strength(df.value, period=period))
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

# Fit ARIMA model
if do_fit:
    period = int(60 / 5)
    auto_fit = True
    if auto_fit:
        arima = auto_arima(
            train,
            stepwise=True,
            trace=1,
            m=period,
            information_criterion="aicc",
            seasonal=False,
            error_action="ignore",
            suppress_warnings=True,
        )
        print(arima.summary())
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arima = ARIMA(order=(4, 1, 4), seasonal_order=None)
            arima.fit(train)

    # Diagnostics plot
    arima.plot_diagnostics(lags=50)
    box_ljung(arima.resid(), nlags=20).format()
    plt.gcf().suptitle('Diagnostics Plot')
    plt.figure()
    plt.plot(df.value.index[:train_len - 1],
             np.abs(arima.resid()), label="abs(residuals)")
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
    plt.figure()
    plt.plot(df.value.index[train_len:], test,
             '--', color='C0', label="test set")
    plt.plot(df.value.index[train_len:], predictions,
             '--', color='C1', label="forecasted values")
    plt.plot(df.value.index[:train_len], train, color='C0', label="train set")
    plt.plot(df.value.index[:train_len - 1], fitted_values,
             color='C1', label="fitted values")
    plt.legend()
    plt.gcf().suptitle("Fitted values and forecasts")

# Evaluate the fit residuals to identify outliers.
if do_evaluate:
    # Re-fit the model on the entire data.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arima = ARIMA(order=(4, 1, 4), seasonal_order=None)
        # arima = ARIMA(order=arima.order, seasonal_order=arima.seasonal_order)
        arima.fit(df.value)

    # Binary search to find optimal thresh
    # contamination = len(labels) / len(labels_windows)
    contamination = 0.015
    expected_anomalies = contamination * len(df.value)
    print("expected_anomalies", expected_anomalies)
    found_anomalies = len(df.value)
    upper = arima.resid().max()
    thresh = upper / 2
    lower = 0

    i = 0
    max_iter = 32
    while i < max_iter:
        pred = np.where(np.abs(arima.resid()) > thresh, -1, 1)
        found_anomalies = (pred == -1).sum()
        error = found_anomalies - expected_anomalies
        if 0 <= error <= 1:
            print("finished")
            break
        if found_anomalies < expected_anomalies:
            # Decrement threshold
            upper = thresh
            thresh = thresh - (thresh - lower) / 2
        else:
            # Increment threshold
            lower = thresh
            thresh = thresh + (upper - thresh) / 2
        i += 1

    # thresh = 7.05362 * (10 ** 6)
    # pred = np.where(np.abs(arima.resid()) > thresh, -1, 1)

    gt_pred, gt_windows = get_gt_arrays(
        df.index, df.index, labels, labels_windows)

    # Compute metrics
    metrics_columns = ["precision", "recall", "f_score", "nab_score"]
    Metrics = collections.namedtuple("Metrics", metrics_columns)

    # for t in range(int(thresh) - 5000000, int(thresh) + 5000000, 1000000):
    for t in np.linspace(0, arima.resid().max(), num=20):
        pred = np.where(np.abs(arima.resid()) > t, -1, 1)
        nab_score = get_nab_score(gt_windows, pred)
        simple_metrics = get_simple_metrics(gt_pred[:-1], pred)
        metrics = simple_metrics + (nab_score,)
        metrics = Metrics(*metrics)
        print("thresh {:.2f}: {}".format(t, metrics))

        anomalies = df.value[:-1][pred == -1]
        ax = _make_plots(df, df, labels, labels_windows, "", anomalies, "")
        ax.set_title("threshold={} - optimal threshold={}".format(t, thresh))

    plt.show()

plt.show()
