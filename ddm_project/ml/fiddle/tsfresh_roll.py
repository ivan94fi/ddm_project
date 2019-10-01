"""
Testing the behaviour of tsfresh feature extraction function.

Trying make_forecasting_frame and extract_features to transform a univariate
time series in a n_measurements x n_features matrix suitable for machine
learning algorithms.
"""
import time

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
from tsfresh import (  # extract_relevant_features,
    extract_features, select_features
)
from tsfresh.utilities.dataframe_functions import (  # , roll_time_series
    impute, make_forecasting_frame
)

from ddm_project.readers.nab_dataset_reader import NABReader

# import statsmodels.api as sm
# import statsmodels.tsa.api as tsa
# from sklearn.preprocessing import (
#     MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
# )

reader = NABReader()
reader.load_data()
reader.load_labels()

dataset_name = "iio_us-east-1_i-a2eb1cd9_NetworkIn.csv"

df = reader.data.get(dataset_name)
labels = reader.labels.get(dataset_name)

# add_lags = False
# if add_lags:
#    max_lags = 10
#    lagged = pd.concat(
#        [df] + [df.shift(i).rename(columns=lambda c: "{}_lag_{}".format(c, i))
#                for i in range(1, max_lags)], axis=1).dropna()


# rolled = roll_time_series(df,
#                          column_id='id',
#                          column_sort='timestamp',
#                          column_kind='kind',
#                          rolling_direction=1,
#                          max_timeshift=2)
scaled_value = MinMaxScaler().fit_transform(
    df.value.values.reshape(-1, 1))[:, 0]
scaled_value = pd.Series(scaled_value, index=df.index, name=df.value.name)

df_shift, y = make_forecasting_frame(scaled_value,
                                     kind="kind",
                                     max_timeshift=10,
                                     rolling_direction=1)

if False:
    extract_start = time.time()
    X = extract_features(df_shift,
                         column_id="id",
                         column_sort="time",
                         column_value="value",
                         impute_function=impute,
                         n_jobs=8,
                         show_warnings=False)
    extract_end = time.time()
    print("Extraction time: {}".format(extract_end - extract_start))
    raw_feat_num = X.shape[1]
    print("Extracted {} features.".format(raw_feat_num))
    X = X.loc[:, X.apply(pd.Series.nunique) != 1]
    print("Dropped {} constant features. Remaining features: {}".format(
        raw_feat_num - X.shape[1], X.shape[1]))
    select_start = time.time()
    X = select_features(X, y, ml_task='regression')
    select_end = time.time()
    print("Selection time: {}".format(select_end - select_start))
    print("Final filtered features: {}".format(X.shape[1]))
    X.to_pickle("X.pkl")
else:
    print("Reading features from file")
    X = pd.read_pickle("X.pkl")

tdf = df[1:]
nu = 0.0015
gamma = 0.02
# X = StandardScaler().fit_transform(X)
predictions = OneClassSVM(nu=nu, gamma='auto').fit_predict(X)
anomalies = tdf.value.mask(predictions == -1)
plt.plot(tdf.index, tdf.value)
plt.plot(anomalies, 'x')
plt.show()
y_gt = pd.Series(1, tdf.index)
y_gt.loc[labels] = -1
print(classification_report(y_gt, predictions, labels=[-1]))
