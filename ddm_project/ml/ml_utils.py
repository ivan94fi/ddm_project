"""Utilities for grid search of machine learning models."""

import os
import pickle

import pandas as pd
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from ddm_project.metrics.metrics import get_nab_score, get_simple_metrics
from ddm_project.utils.utils import _format_parameters


class PredictionsGenerator(object):
    """Class to perform grid search on machine learning models."""

    def __init__(self, model_class, model_name, parameters, dataset_name,
                 directory="predictions", filename=None):
        self.model_class = model_class
        self.model_name = model_name
        self.parameters = parameters
        self.dataset_name = dataset_name
        self.parameters_grid = ParameterGrid(parameters)
        self.predictions_grid = {}
        self.directory = directory
        if filename is None:
            self.filename = self.dataset_name[:-4]\
                + "_" + self.model_name + "_predictions.pkl"
        else:
            self.filename = filename
        self.filepath = os.path.join(self.directory, self.filename)

    def get(self, X, read=True):
        """Obtain predictions for each parameter in the grid."""
        needs_computing = False
        if read and self.predictions_grid == {}:
            if not self.read_from_disk():
                tqdm.write("File not found, recreating predictions")
                needs_computing = True
        else:
            needs_computing = True
        if needs_computing:
            desc = "{:>8}".format(self.model_name) + " fit and predict"
            for params in tqdm(self.parameters_grid, desc=desc):
                # construct a model instance
                model = self.model_class(**params)
                pred = model.fit_predict(X)
                param_str = _format_parameters(params)
                self.predictions_grid[param_str] = pred
            self.save_on_disk()

        return self.predictions_grid

    def save_on_disk(self):
        """Save predictions on disk."""
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)
        with open(self.filepath, "wb") as f:
            pickle.dump(self.predictions_grid, f)
        tqdm.write("Saving predictions on disk")

    def read_from_disk(self):
        """Read predictions from disk."""
        if not os.path.isfile(self.filepath):
            return False
        with open(self.filepath, "rb") as f:
            self.predictions_grid = pickle.load(f)
        tqdm.write("Reading predictions from disk")
        return True


class MetricsGenerator(object):
    """Compute metrics from predictions grid."""

    def __init__(self, model_name, dataset_name,
                 directory="metrics", filename=None):
        self.directory = directory
        self.filename = filename
        self.model_name = model_name
        if filename is None:
            self.filename = dataset_name[:-4] + \
                "_" + self.model_name + "_metrics.pkl"
        else:
            self.filename = filename
        self.filepath = os.path.join(self.directory, self.filename)
        self.metrics_columns = ["precision", "recall", "f_score", "nab_score"]
        self.metrics = None

    def get(self, predictions, gt_pred, gt_windows, read=True):
        """Compute or read metrics."""
        needs_computing = False
        if read and self.metrics is None:
            if not self.read_from_disk():
                tqdm.write("File not found, recreating metrics")
                needs_computing = True
        else:
            needs_computing = True
        if needs_computing:
            metrics = {}
            for param_str, pred in predictions.items():
                nab_score = get_nab_score(gt_windows, pred)
                metrics[param_str] = get_simple_metrics(
                    gt_pred, pred) + (nab_score,)
            self.metrics = pd.DataFrame(
                metrics, index=self.metrics_columns).T
            self.save_on_disk()

        return self.metrics

    def save_on_disk(self):
        """Save metrics on disk."""
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)
        with open(self.filepath, "wb") as f:
            pickle.dump(self.metrics, f)
        tqdm.write("Saving metrics on disk")

    def read_from_disk(self):
        """Read metrics from disk."""
        if not os.path.isfile(self.filepath):
            return False
        with open(self.filepath, "rb") as f:
            self.metrics = pickle.load(f)
        tqdm.write("Reading metrics from disk")
        return True


if __name__ == '__main__':
    from pprint import pprint
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import OneClassSVM
    from sklearn.decomposition import PCA
    from ddm_project.readers.nab_dataset_reader import NABReader
    from ddm_project.utils.utils import get_gt_arrays
    from ddm_project.ml.feature_generation import FeatureGenerator

    dataset_name = "iio_us-east-1_i-a2eb1cd9_NetworkIn.csv"
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

    ocsvm_params = {'nu': [0.001], 'gamma': [0.001]}

    predictions_generator = PredictionsGenerator(
        OneClassSVM, "ocsvm", ocsvm_params, dataset_name)
    predictions_generator.get(X, read=True)

    pprint(predictions_generator.predictions_grid)

    tdf = df.loc[X.index, :]  # Truncated df. Includes only the usable indexes
    gt_pred, gt_windows = get_gt_arrays(
        tdf.index, df.index, labels, labels_windows)

    metrics_generator = MetricsGenerator("ocsvm", dataset_name)
    metrics_generator.get(
        predictions_generator.predictions_grid, gt_pred, gt_windows)

    pprint(metrics_generator.metrics)
