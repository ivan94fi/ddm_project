"""
Example script to apply Isolation Forests for anomaly detection to NAB dataset.

The scripts does the following actions:
    * Read dataset.
    * Compute features and apply PCA.
    * Fit several IsolationForest models with different parameters.
    * Plot the anomalies found for a qualitative evaluation.
    * Calculate precision, recall and F-score for each method, for a
      quantitative evaluation.
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from ddm_project.ml.feature_generation import FeatureGenerator
from ddm_project.readers.nab_dataset_reader import NABReader

register_matplotlib_converters()

seed = 42

reader = NABReader()
reader.load_data()
reader.load_labels()

dataset_name = "iio_us-east-1_i-a2eb1cd9_NetworkIn.csv"

df = reader.data.get(dataset_name)
labels = reader.labels.get(dataset_name)

scaler = StandardScaler()

feature_generator = FeatureGenerator(
    df, ts_col='value', scaler=scaler)
X = feature_generator.get(read=True)

pca = PCA(n_components=50)
X = pd.DataFrame(pca.fit_transform(X), index=X.index)


# param_grid = {'nu': [0.001, 0.0015, 0.002, 0.003, 0.005, 0.01]}
param_grid = {'nu': np.arange(0, 0.1, 0.005)[1:]}

grid = ParameterGrid(param_grid)

pred_list = []
for params in tqdm(grid, desc="Fitting models and predicting"):
    contamination = params['nu']
    model = IsolationForest(contamination=contamination,
                            random_state=seed, behaviour='new')
    pred = model.fit_predict(X)
    pred_list.append((pred, params))


files = glob.glob('tmp/*.png')
for f in files:
    os.remove(f)

tdf = df.loc[X.index, :]
y_gt = pd.Series(1, tdf.index)
y_gt.loc[labels] = -1

results = []
for pred, params in tqdm(pred_list):
    plt.plot(df.index, df.value, label='Original data',
             linestyle='--', alpha=0.5)
    plt.plot(tdf.index, tdf.value, label='Used data', color='C0')
    anomalies = tdf.value[pred == -1]

    plt.plot(anomalies, 'x', label="Predicted anomalies", markersize=10)
    plt.plot(labels, tdf.loc[labels], 'o',
             markersize=5, label="Real anomalies")
    title_str = "nu_{:.3f}".format(params["nu"])  # 'nu_' + str(params['nu'])
    plt.title(title_str)
    plt.legend()
    plt.show()
    # plt.savefig("tmp/" + title_str + ".png")
    # plt.clf()

    results.append(precision_recall_fscore_support(
        y_gt, pred, beta=1.0, labels=[-1]))

for res in results:
    # print("precision: {:.4f} {:.4f} recall: {:.4f} {:.4f} F1:
    #       {:.4f} {:.4f}".format(
    #     res[0][0], res[0][1], res[1][0], res[1][1], res[2][0], res[2][1]))
    print("precision: {:.4f} recall: {:.4f} F1: {:.4f}".format(
        res[0][0], res[1][0], res[2][0]))
