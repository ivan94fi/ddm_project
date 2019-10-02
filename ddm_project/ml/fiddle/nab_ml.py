"""
Example script to apply machine learning techinques for AD to NAB dataset.

The scripts does the following actions:
    * Read dataset.
    * Compute features and apply PCA.
    * Fit several OneClassSVM models with different parameters.
    * Plot the anomalies found for a qualitative evaluation.
    * Calculate precision, recall and F-score for each method, for a
      quantitative evaluation.
    * Calculate NAB score.
"""

import collections
import warnings

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

deprecation_message = "This script is outdated. The new version is the "\
                      + "script `grid_search_ml.py` in the `ml` package"

warnings.warn(deprecation_message, DeprecationWarning)

seed = 42
use_iforest = True
use_ocsvm = True
iforest_name = 'iforest'
ocsvm_name = 'ocsvm'
models_to_use = []
if use_iforest:
    models_to_use.append(iforest_name)
if use_ocsvm:
    models_to_use.append(ocsvm_name)
if not models_to_use:
    raise ValueError("At least a model must be used.")


def _fmt_param(d):
    """Format a dictionary of parameter to a string.

    Parameters
    ----------
    d : dict
        The dictionary to format.

    Returns
    -------
    str
        A string representation of input, useful as a key for dictionaries and
        as title in plots.

    Example: {"param1": 0.01, "param2": 67} -> "param1_0.01__param2_67"

    """
    finalstr = ""
    for k, v in d.items():
        if k == 'behaviour':
            continue
        if type(v) == str:
            pstr = str(k) + '_' + str(v)
        else:
            pstr = "{}_{:.4f}".format(k, v)
        finalstr = finalstr + pstr + "__"
    return finalstr.strip('_')


register_matplotlib_converters()

reader = NABReader()
reader.load_data()
reader.load_labels()

# dataset_name = "ec2_cpu_utilization_fe7f93.csv"
# dataset_name = "rds_cpu_utilization_e47b3b.csv"
# dataset_name = "grok_asg_anomaly.csv"
# dataset_name = "elb_request_count_8c0756.csv"
# dataset_name = "iio_us-east-1_i-a2eb1cd9_NetworkIn.csv"
dataset_name = "machine_temperature_system_failure.csv"

df = reader.data.get(dataset_name)
labels = reader.labels.get(dataset_name)
labels_windows = reader.label_windows.get(dataset_name)

scaler = StandardScaler()

feature_generator = FeatureGenerator(
    df, ts_col='value', scaler=scaler,
    fname=dataset_name.replace(".csv", ".pkl")
)
X = feature_generator.get(read=True)

pca = PCA(n_components=50)
X = pd.DataFrame(pca.fit_transform(X), index=X.index)

raise SystemExit("Interrupted")
models_classes = {iforest_name: IsolationForest, ocsvm_name: OneClassSVM}

# Parameters for grid search
iforest_params = {'contamination': np.arange(0, 0.1, 0.005)[1:],
                  'behaviour': ['new']}
ocsvm_params = {'nu': [0.001, 0.0015, 0.002, 0.003, 0.005, 0.01],
                'gamma': ['scale', 0.0005, 0.001, 0.0025, 0.005, 0.01]}

# ocsvm_params = {'nu': [0.001], 'gamma': [0.001]}

params_dict = {iforest_name: iforest_params, ocsvm_name: ocsvm_params}
param_grids = {name: ParameterGrid(p) for name, p in params_dict.items()}

Result = collections.namedtuple('Result', 'model pred')
predictions = {n: {} for n in models_to_use}
for model_name in models_to_use:
    model_class = models_classes[model_name]
    desc = "{:>8}".format(model_name) + " fit and predict"
    for param in tqdm(param_grids[model_name], desc=desc):
        model = model_class(**param)
        pred = model.fit_predict(X)
        param_str = _fmt_param(param)
        predictions[model_name][param_str] = Result(model=model, pred=pred)


tdf = df.loc[X.index, :]
gt_pred = pd.Series(1, tdf.index)
gt_pred.loc[labels] = -1

gt_windows = []
idf = pd.DataFrame(index=df.index)
idf['idx'] = idf.reset_index().index

for win in labels_windows:
    win_start = idf.idx.at[win[0]]
    win_end = idf.idx.at[win[1]]
    gt_windows.append((win_start, win_end))


def _make_plots(model, anomalies, params):
    plt.plot(df.index, df.value, label='Original data',
             linestyle='--', alpha=0.5)
    plt.plot(tdf.index, tdf.value, label='Used data', color='C0')
    plt.plot(anomalies, 'x', label="Predicted anomalies")
    plt.plot(labels, tdf.loc[labels], 'o',
             markersize=5, label="Real anomalies")
    y_min, y_max = plt.ylim()
    for win in labels_windows:
        plt.fill_between(win, y_min, y_max, color='r', alpha=0.1)
        # plt.plot(win, tdf.loc[win],
        #          '^', markersize=5, label="anomalies windows")
    plt.ylim(y_min, y_max)
    plt.title(model + " " + params)
    plt.legend()


plot = False
# if plot:
#     files = glob.glob('tmp/*.png')
#     for f in files:
#         os.remove(f)
columns = ["precision", "recall", "f_score", "nab_score"]
metrics = {n: {} for n in models_to_use}
for model, data in predictions.items():
    curr_metrics = {}
    for params, result in data.items():
        anomalies = tdf.value[result.pred == -1]
        if plot:
            _make_plots(model, anomalies, params)
            plt.show()
            # plt.savefig("tmp/" + title_str + ".png"); plt.clf()
        nab_score = get_nab_score(gt_windows, result.pred)
        curr_metrics[params] = get_simple_metrics(gt_pred, result.pred)\
            + (nab_score,)

    metrics[model] = pd.DataFrame(curr_metrics, index=columns).T


for k, v in metrics.items():
    v['nab_score'] = (v['nab_score'] - v['nab_score'].mean()
                      ) / v['nab_score'].std()
    plt.figure()
    plt.xticks(rotation=30, ha='right', size='xx-small')
    for c in columns:
        plt.plot(c, data=v)
    plt.legend()
    plt.title(k)

"""
Da qui si vede che il parametro migliore per la contaminazione è 0.02
(ci sta perché sono 0.015% le anomalie)

Gamma per ocsvm è 0.05 mi pare.
"""

plt.show()

# TODO: Plot scores for IForest model
# iforest_scores = iforest_model.decision_function(X_shuffle)
# plt.plot(iforest_scores)
# idx = np.where(iforest_pred == -1)[0]
# plt.plot(idx, iforest_scores[idx], 'x')
# plt.title("Isolation forest anomaly scores")
# plt.show()
