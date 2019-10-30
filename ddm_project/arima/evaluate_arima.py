"""Module to evaluate ARIMA models."""

import argparse
import collections
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import register_matplotlib_converters
from pmdarima.arima import ARIMA

from ddm_project.metrics.metrics import get_nab_score, get_simple_metrics
from ddm_project.readers.nab_dataset_reader import NABReader
from ddm_project.utils.utils import get_gt_arrays, make_predictions_plots


def get_arima_predictions(
    residuals, window_size, alpha, weighted=True, stddev_coeff=2
):
    """Compute predictions using rolling window strategy."""
    pred = np.zeros_like(residuals)
    left = 0
    right = 0
    i = 0
    while right < len(residuals):
        window = residuals[left: right + 1].copy()
        if False:  # i < window_size:
            # print("window:", window)
            debug_arr = np.zeros(residuals.shape[0])
            debug_arr[left: right + 1] = 1
            print(debug_arr.astype(np.int))

        if weighted:
            weights = (1 - alpha) ** np.arange(window.shape[0])
            weights = np.flip(weights / weights.sum())
            window_mean = np.dot(window, weights)
            window_std = np.sqrt(np.dot(weights, (window - window_mean) ** 2))
        else:
            window_mean = window.mean()
            window_std = window.std()

        if np.abs(residuals[i] - window_mean) > stddev_coeff * window_std:
            prediction = -1
        else:
            prediction = 1
        pred[i] = prediction
        if right + 1 < window_size:
            left = left
            right += 1
        else:
            left += 1
            right += 1
        i += 1
    return pred


if __name__ == "__main__":
    register_matplotlib_converters()

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_index", type=int, choices=range(6))
    args = parser.parse_args()

    dataset_names = [
        "iio_us-east-1_i-a2eb1cd9_NetworkIn.csv",
        "machine_temperature_system_failure.csv",
        "ec2_cpu_utilization_fe7f93.csv",
        "rds_cpu_utilization_e47b3b.csv",
        "grok_asg_anomaly.csv",
        "elb_request_count_8c0756.csv",
    ]

    dataset_name = dataset_names[args.dataset_index]
    print("Chosen dataset:", dataset_name)

    reader = NABReader()
    reader.load_data()
    reader.load_labels()

    # Load data and labels
    df = reader.data.get(dataset_name)
    labels = reader.labels.get(dataset_name)
    labels_windows = reader.label_windows.get(dataset_name)

    # Evaluate the fit residuals to identify outliers.
    whole_df = df.copy()
    df = df.iloc[10:, :]

    fname = "arima.pkl"
    if not os.path.isfile(fname):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arima = ARIMA(order=(4, 1, 4), seasonal_order=None)
            # arima = ARIMA(
            #     order=arima.order, seasonal_order=arima.seasonal_order)
            arima.fit(df.value)
            joblib.dump(arima, fname, compress=3)
    else:
        arima = joblib.load(fname)

    gt_pred, gt_windows = get_gt_arrays(
        df.index, df.index, labels, labels_windows
    )

    # Compute metrics
    metrics_columns = ["precision", "recall", "f_score", "nab_score"]
    Metrics = collections.namedtuple("Metrics", metrics_columns)

    window_size = 30
    alpha = 0.15
    pred = get_arima_predictions(arima.resid(), window_size, alpha)
    print("Anomalies number:", pred[pred == -1].shape[0])

    nab_score = get_nab_score(gt_windows, pred)
    simple_metrics = get_simple_metrics(gt_pred[1:], pred)
    metrics = simple_metrics + (nab_score,)
    metrics = Metrics(*metrics)

    anomalies = df.value[1:][pred == -1]
    ax = make_predictions_plots(
        whole_df, df, labels, labels_windows, "", anomalies, ""
    )
    line = "& {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\".format(
        "ARIMA",
        metrics.nab_score,
        metrics.f_score,
        metrics.precision,
        metrics.recall,
    )
    print(line)

    plt.show()
