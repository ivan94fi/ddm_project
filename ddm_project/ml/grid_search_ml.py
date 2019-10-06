"""
This script encapsulates the process of grid search for the ml techinques.

After retrieving (or extracting) the features, a predefined set of models is
fitted. The obtained models are used to generate predictions and these
predictions are evaluated against the ground truth labels for anomalies.

Plots are generated to report the positions of anomalies as defined by the
various models, and to show the evolution of the computed metrics as the
model parameters change.
"""

# TODO: separare le fasi dello script, magari in classi diverse.
# TODO: refactor parte iniziale su definizione dei parametri.

import argparse
import collections
import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from tqdm import tqdm

from ddm_project.metrics.metrics import get_nab_score, get_simple_metrics
from ddm_project.ml.feature_generation import FeatureGenerator
from ddm_project.readers.nab_dataset_reader import NABReader
from ddm_project.utils.utils import (
    _format_parameters, _make_plots, get_gt_arrays
)

Result = collections.namedtuple('Result', ["model", "pred"])

logger = logging.getLogger(__name__)

# TODO: separare le due cose.
logger.warning(
    "Ad ora se si leggono le metriche non si possono fare i plot con le "
    "predizioni. Bisognerebbe dividere le due cose e fare in modo da poter "
    "usare le pred anche se non si calcolano le metriche."
)

parser = argparse.ArgumentParser()
parser.add_argument("dataset_index", type=int, choices=range(6))
parser.add_argument("--iforest", action="store_true")
parser.add_argument("--ocsvm", action="store_true")
args = parser.parse_args()

dataset_names = ["iio_us-east-1_i-a2eb1cd9_NetworkIn.csv",
                 "machine_temperature_system_failure.csv",
                 "ec2_cpu_utilization_fe7f93.csv",
                 "rds_cpu_utilization_e47b3b.csv",
                 "grok_asg_anomaly.csv",
                 "elb_request_count_8c0756.csv"]

dataset_name = dataset_names[args.dataset_index]
print("Chosen dataset:", dataset_name)
iforest_name = 'iforest'
ocsvm_name = 'ocsvm'
models_to_use = []
if args.iforest:
    models_to_use.append(iforest_name)
if args.ocsvm:
    models_to_use.append(ocsvm_name)
if not models_to_use:
    raise ValueError("At least a model must be used.")
models_classes = {iforest_name: IsolationForest, ocsvm_name: OneClassSVM}

metrics_columns = ["precision", "recall", "f_score", "nab_score"]

register_matplotlib_converters()

reader = NABReader()
reader.load_data()
reader.load_labels()

# Get dataset and labels
df = reader.data.get(dataset_name)
labels = reader.labels.get(dataset_name)
labels_windows = reader.label_windows.get(dataset_name)

# Retrieve / compute features
scaler = StandardScaler()
feature_generator = FeatureGenerator(
    df, ts_col='value', scaler=scaler,
    fname=dataset_name.replace(".csv", ".pkl")
)
X = feature_generator.get(read=True)

# Apply dimensionality reduction through PCA
pca = PCA(n_components=50)
X = pd.DataFrame(pca.fit_transform(X), index=X.index)

# Construct ground truth arrays from labels and window labels
tdf = df.loc[X.index, :]  # Truncated df. Includes only the usable indexes
gt_pred, gt_windows = get_gt_arrays(
    tdf.index, df.index, labels, labels_windows)

read_metrics = True
needs_computing = True
metrics_fname = "metrics/" + dataset_name[:-4] + "_metrics.pkl"
if read_metrics:
    try:
        with open(metrics_fname, "rb") as f:
            metrics = pickle.load(f)
        needs_computing = False
        print("Metrics read from disk")
    except FileNotFoundError:
        print("Metrics not found on disk. Recomputing.")
        needs_computing = True

if needs_computing:
    # Parameters for grid search
    iforest_params = {
        'contamination': np.arange(0, 0.1, 0.005)[1:],
        'behaviour': ['new']}
    ocsvm_params = {
        'nu': [0.001, 0.0015, 0.002, 0.003, 0.005, 0.01],
        'gamma': ['scale', 0.0005, 0.001, 0.0025, 0.005, 0.01]}
    # ocsvm_params = {'nu': [0.001], 'gamma': [0.001]}
    params_dict = {iforest_name: iforest_params, ocsvm_name: ocsvm_params}
    param_grids = {name: ParameterGrid(p) for name, p in params_dict.items()}

    # Fit and predict models
    # predictions is a dict: its keys are model classes, its values are dict
    # These inner dicts have parameter formatted as strings for keys and Result
    # as values.
    predictions = {n: {} for n in models_to_use}
    # For each model, compute predictions for every possible combination of its
    # parameters
    for model_name in models_to_use:
        model_class = models_classes[model_name]
        desc = "{:>8}".format(model_name) + " fit and predict"
        for params in tqdm(param_grids[model_name], desc=desc):
            model = model_class(**params)  # construct a model instance
            pred = model.fit_predict(X)
            param_str = _format_parameters(params)
            predictions[model_name][param_str] = Result(model=model, pred=pred)

    """
    Example of predictions:
    predictions = {
        "iforest": {
            "contamination__0.001": Result(fitted_model, prediction_array),
            "contamination__0.005": Result(fitted_model, prediction_array),
            # ...
        },
        "ocsvm": {
            "gamma_0.1__nu_0.02": Result(fitted_model, prediction_array),
            "gamma_0.1__nu_0.05": Result(fitted_model, prediction_array),
            # ...
        }
    }
    """

    print("Computing metrics")
    # Compute metrics and save them
    metrics = {}  # n: {} for n in models_to_use
    # metrics is a dict of dataframes containing metrics for both models.
    # E.g. metrics["ocsvm"] is a dataframe with params string on rows and
    # columns for each of the metrics
    for model_name, data in predictions.items():
        curr_metrics = {}
        for param_str, result in data.items():
            nab_score = get_nab_score(gt_windows, result.pred)
            curr_metrics[param_str] = get_simple_metrics(
                gt_pred, result.pred) + (nab_score,)
        metrics[model_name] = pd.DataFrame(
            curr_metrics, index=metrics_columns).T

    do_write = True
    if os.path.isfile(metrics_fname):
        overwrite = input("Metrics file already exists. Overwrite?[y/n] ")
        if overwrite.lower() != "y":
            do_write = False

    if do_write:
        with open(metrics_fname, "wb") as f:
            pickle.dump(metrics, f)
        print("Metrics written to disk")

    # Plot original data + predicted anomalies and anomaly windows.
    plot_pred_gt_anomalies = False
    # TODO: questo andrebbe staccato e messo in un altro script
    if plot_pred_gt_anomalies:
        # files = glob.glob('tmp/*.png')
        # for f in files:
        #     os.remove(f)
        for model_name, data in predictions.items():
            for param_str, result in data.items():
                anomalies = tdf.value[result.pred == -1]
                # Plot original data with gt anomaly windows and
                # found anomalies
                ax = _make_plots(df, tdf, labels, labels_windows,
                                 model_name, anomalies, param_str)
                plt.show()
                # plt.savefig("tmp/" + model + " " + param_str + ".png")
                # plt.clf()

# Plot the evolutions of metrics with different parameters for both models.
for model_name, model_metrics in metrics.items():
    nab_score = model_metrics['nab_score']
    std_nab_score = (nab_score - nab_score.mean()) / nab_score.std()
    model_metrics['nab_score'] = std_nab_score
    plt.figure(figsize=(9.6, 6.0))
    # plt.figure()
    plt.xticks(rotation=30, ha='right', size='xx-small')
    for c in metrics_columns:
        plt.plot(c, data=model_metrics)
    plt.grid(axis='x', color='gray', linestyle='--', linewidth=1, alpha=0.5)
    plt.legend()
    fig_title = dataset_name[:-4] + ": " + model_name
    plt.title(fig_title)
    plt.tight_layout()
    fig_fname = "metrics/" + dataset_name[:-4] + "_" + model_name + ".svg"
    plt.savefig(fig_fname)


plt.show()

# TODO: Plot scores for IForest model
# iforest_scores = iforest_model.decision_function(X_shuffle)
# plt.plot(iforest_scores)
# idx = np.where(iforest_pred == -1)[0]
# plt.plot(idx, iforest_scores[idx], 'x')
# plt.title("Isolation forest anomaly scores")
# plt.show()
