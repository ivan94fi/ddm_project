"""
This script encapsulates the process of grid search for the ml techinques.

After retrieving (or extracting) the features, a predefined set of models is
fitted. The obtained models are used to generate predictions and these
predictions are evaluated against the ground truth labels for anomalies.

Plots are generated to report the positions of anomalies as defined by the
various models, and to show the evolution of the computed metrics as the
model parameters change.
"""

import argparse
import itertools

import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from ddm_project.ml.feature_generation import FeatureGenerator
from ddm_project.ml.ml_utils import MetricsGenerator, PredictionsGenerator
from ddm_project.readers.nab_dataset_reader import NABReader
from ddm_project.utils.utils import get_gt_arrays, make_predictions_plots

parser = argparse.ArgumentParser()
parser.add_argument("dataset_index", type=int, choices=range(6))
parser.add_argument("-i", "--iforest", action="store_true")
parser.add_argument("-o", "--ocsvm", action="store_true")
parser.add_argument("--plot_predictions", action="store_true")
parser.add_argument("--plot_metrics", action="store_true")
parser.add_argument("--test_params", action="store_true")
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

register_matplotlib_converters()
lw = 1
fs = 7
mp.rcParams["font.size"] = fs
mp.rcParams["axes.linewidth"] = lw
mp.rcParams["lines.linewidth"] = lw
mp.rcParams["patch.linewidth"] = lw
mp.rcParams["font.family"] = "serif"

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
    tdf.index, df.index, labels, labels_windows
)

# Parameters for grid search
if args.test_params:
    iforest_params = {"contamination": [0.001], 'behaviour': ['new']}
    ocsvm_params = {'nu': [0.001], 'gamma': [0.001]}
else:
    iforest_params = {
        'contamination': np.arange(0, 0.1, 0.005)[1:],
        'behaviour': ['new']
    }
    ocsvm_params = {
        'nu': [0.001, 0.0015, 0.002, 0.003, 0.005, 0.01],
        'gamma': ['scale', 0.0005, 0.001, 0.0025, 0.005, 0.01]
    }
params_dict = {iforest_name: iforest_params, ocsvm_name: ocsvm_params}

# Perform grid search on the chosen models.
predictions = {}
metrics = {}
for model_name in models_to_use:
    # model_class, model_name, parameters, dataset_name,
    predictions_generator = PredictionsGenerator(
        models_classes[model_name],
        model_name,
        params_dict[model_name],
        dataset_name
    )
    predictions[model_name] = predictions_generator.get(X, read=True)

    metrics_generator = MetricsGenerator(model_name, dataset_name)
    metrics[model_name] = metrics_generator.get(
        predictions[model_name], gt_pred, gt_windows)

# Plot original data with gt anomaly windows and found anomalies
if args.plot_predictions:
    for model_name, model_predictions in predictions.items():
        for param_str, pred in model_predictions.items():
            anomalies = tdf.value[pred == -1]
            ax = make_predictions_plots(df, tdf, labels, labels_windows,
                                        model_name, anomalies, param_str)
            plt.show()

# Plot the evolutions of metrics with different parameters for both models.
if args.plot_metrics:
    print(dataset_name.replace("_", "\\_"))
    for model_name, model_metrics in metrics.items():
        mx = model_metrics.max()
        if model_name == "iforest":
            name = "iForest"
        else:
            name = "OCSVM"
        line = "& {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\".format(
            name, mx.nab_score, mx.f_score, mx.precision, mx.recall)
        print(line)

    for model_name, model_metrics in metrics.items():
        fig, ax = plt.subplots()  # figsize=(9.6, 6.0)
        ax.tick_params(axis='x', labelrotation=30, labelsize=7)
        lines = []
        for c in model_metrics.columns:
            if c != "nab_score":
                line = ax.plot(c, data=model_metrics)
                lines.append(line)
        twin_ax = ax.twinx()
        line = twin_ax.plot("nab_score", data=model_metrics, color="C3")
        lines.append(line)
        lines = list(itertools.chain.from_iterable(lines))
        ax.legend(lines, [l.get_label() for l in lines])
        ax.grid(axis='x', color='gray',
                linestyle='--', linewidth=1, alpha=0.5)
        fig_title = dataset_name[:-4] + ": " + model_name
        ax.set_title(fig_title)
        # fig_fname = "metrics/{}_{}.svg".format(dataset_name[:-4], model_name)
        # plt.savefig(fig_fname)
plt.show()
