"""
try_nab_ml.py.

Example script to apply ml techinques to NAB dataset.
"""
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

from features_generation import FeatureGenerator
from nab_dataset_reader import NABReader

# from sklearn.metrics import classification_report


register_matplotlib_converters()

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

#####################
# raise InterruptedError("*" * 35 + " Manually interrupted " + "*" * 35)
#####################


# Fit One-Class SVM
contamination = 0.002
gamma = 0.02
seed = 42

param_grid = {'nu': [0.001, 0.0015, 0.002, 0.003, 0.005, 0.01],
              'gamma': ['scale', 0.0005, 0.001, 0.0025, 0.005, 0.01]}
# param_grid = {'nu':  np.arange(0.0001, 0.01, 0.0005),
#               'gamma': np.arange(0.0005, 0.01, 0.001)}
# param_grid = {'nu': [0.0015, 0.03], 'gamma': [0.2, 0.3]}
grid = ParameterGrid(param_grid)

pred_list = []
for params in tqdm(grid):
    contamination = params['nu']
    gamma = params['gamma']
    ocsvm_model = OneClassSVM(**params)
    ocsvm_pred = ocsvm_model.fit_predict(X)
    pred_list.append((ocsvm_pred, params))

for pred, params in tqdm(pred_list):
    plt.plot(df.index, df.value, label='Original data',
             linestyle='--', alpha=0.5)
    tdf = df.loc[X.index, :]
    plt.plot(tdf.index, tdf.value, label='Used data', color='C0')
    anomalies = tdf.value[pred == -1]

    plt.plot(anomalies, 'x', label="Predicted anomalies", markersize=10)
    plt.plot(labels, tdf.loc[labels], 'o',
             markersize=5, label="Real anomalies")
    title_str = 'nu_' + str(params['nu']) + '__gamma_' + str(params['gamma'])
    plt.title(title_str)
    plt.legend()
    # plt.show()
    plt.savefig("tmp/" + title_str + ".png")
    plt.clf()
