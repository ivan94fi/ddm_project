"""
This module defines a class used to generate features from time-series.

The generated features include time-lags up to a certain order and features
extracted with the `tsfresh` package. TODO: use featuretools to extract
temporal features.

The features are saved on disk after the first computation for efficiency.
"""

import os
import time

import pandas as pd
from tqdm import tqdm
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import (
    impute, make_forecasting_frame
)


class FeatureGenerator(object):
    """Encapsulate the feature generation process with `tsfresh`.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the column representing values of
        the time-series.
    ts_col : str
        Name of the column of the time series.
    use_lags : bool
         Include lag features (the default is True). The lags number is
         controlled by `max_timeshift` parameter.
    use_tsfresh : bool
         Include automatic feature generation with `tsfresh`
         package (the default is True).
    max_timeshift : int
         Number of lag features to utilize (the default is 10).
    fname : str
        Filename for features reading and writing (the default is
        'X_gen.pkl').
    scaler : sklearn.TransformerMixin
        perform this scaling after computing features (the default
        is None). The instance must implement the fit_transform method

    Attributes
    ----------
    features : pandas.DataFrame
        The computed features are stored in this attribute.
    features_dir: str
        Directory for storing the computed features.
    path: str
        Full filename of the generated features pickle file.

    """

    def __init__(self, df, ts_col,
                 use_lags=True, use_tsfresh=True,
                 max_timeshift=10, fname='X_gen.pkl', scaler=None):
        self.df = df
        self.ts_col = ts_col
        self.features = None
        self.use_lags = use_lags
        self.use_tsfresh = use_tsfresh
        self.max_timeshift = max_timeshift
        self.fname = fname
        self.scaler = scaler
        self.features_dir = 'generated_features'
        self.path = os.path.join(self.features_dir, self.fname)

    def get(self, read=True):
        """Compute or read features from file.

        Parameters
        ----------
        read : bool
            Read the features from disk, if possible (the default is True).
        Returns
        -------
        pandas.DataFrame
            The computed features.

        """
        needs_computing = False
        if read and self.features is None:
            if not self.read_from_disk():
                tqdm.write("File not found, recreating features")
                needs_computing = True
        else:
            needs_computing = True

        if needs_computing:
            self.compute()

        return self.features

    def compute(self):
        tqdm.write("computing features")
        if not self.use_lags and not self.use_tsfresh:
            raise ValueError(
                "At least one of use_lags and use_tsfresh must be true")
        self.features = None
        X_gen = None
        X_lag = None
        if self.use_tsfresh:
            X_gen = self.compute_tsfresh_features()
        if self.use_lags:
            X_lag = self.compute_lag_features()

        X = pd.concat([X_gen, X_lag], axis=1).dropna()

        if self.scaler:
            X.loc[:] = self.scaler.fit_transform(X)
            # X = pd.DataFrame(
            #     self.scaler.fit_transform(X),
            #     columns=X.columns,
            #     index=X.index)

        self.features = X
        self.save_on_disk()

    def compute_lag_features(self):
        value = self.df[self.ts_col]
        shifts = range(1, self.max_timeshift)
        lags = [value.shift(i).rename("{}_lag_{}".format(self.ts_col, i))
                for i in shifts]
        X_lag = pd.concat(lags, axis=1).dropna()

        return X_lag

    def compute_tsfresh_features(self):
        """Calculate the features using `tsfresh`."""
        value = self.df[self.ts_col]
        df_shift, y = make_forecasting_frame(value,
                                             kind="kind",
                                             max_timeshift=self.max_timeshift,
                                             rolling_direction=1)

        extract_start = time.time()
        X_gen_raw = extract_features(df_shift,
                                     column_id="id",
                                     column_sort="time",
                                     column_value="value",
                                     impute_function=impute,
                                     n_jobs=8,
                                     show_warnings=False)
        extract_end = time.time()
        tqdm.write("Extraction time: {}".format(extract_end - extract_start))

        non_const_idx = X_gen_raw.apply(pd.Series.nunique) != 1
        X_gen_raw_non_const = X_gen_raw.loc[:, non_const_idx]
        select_start = time.time()
        X_gen = select_features(
            X_gen_raw_non_const, y, ml_task='regression')
        select_end = time.time()

        tqdm.write("Filtering time: {}".format(select_end - select_start))
        tqdm.write("Raw features: {}".format(X_gen_raw.shape[1]))
        tqdm.write(
            "Non-constant features: {}".format(X_gen_raw_non_const.shape[1]))
        tqdm.write("Final filtered features: {}".format(X_gen.shape[1]))

        return X_gen

    def save_on_disk(self):
        """Save the features in a Pickle file."""
        self.features.to_pickle(self.path)

    def read_from_disk(self):
        """Read the features from file.

        Returns
        -------
        bool
            True if file is found, False otherwise.

        """
        if not os.path.isfile(self.path):
            return False
        self.features = pd.read_pickle(self.path)
        tqdm.write("Reading features from disk")
        return True


if __name__ == '__main__':
    from nab_dataset_reader import NABReader
    from sklearn.preprocessing import StandardScaler

    reader = NABReader()
    reader.load_data()
    reader.load_labels()

    dataset_name = "iio_us-east-1_i-a2eb1cd9_NetworkIn.csv"

    df = reader.data.get(dataset_name)
    labels = reader.labels.get(dataset_name)
    scaler = StandardScaler()

    feature_generator = FeatureGenerator(
        df, ts_col='value', scaler=scaler, use_tsfresh=False)
    X = feature_generator.get(read=False)

    # lags = feature_generator._lags
