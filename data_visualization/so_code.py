from __future__ import print_function

import numpy as np
import pandas as pd
from pandas.compat import StringIO

data = StringIO("""
timestamp,value
2014-04-02 14:29:00,42.652
2014-04-02 14:34:00,41.361
2014-04-02 14:39:00,-68.408
2014-04-02 14:44:00,40.262
2014-04-02 14:49:00,40.328
2014-04-02 14:54:00,42.652
2014-04-02 14:59:00,-89.836
2014-04-02 15:04:00,42.579
""")

anomalies = ['2014-04-02 14:39:00', '2014-04-02 14:59:00']
anomalies = [pd.to_datetime(a) for a in anomalies]
# anomalies = pd.to_datetime(anomalies)
# print(anomalies.__class__)
df = pd.read_csv(data, parse_dates=['timestamp'], index_col='timestamp')

# Works
# print(df.loc[anomalies[0]])
# print(df.loc[anomalies[1]])

# Works
anomalies_indexes = [np.argwhere(df.index == a).item() for a in anomalies]
# print(df.iloc[anomalies_indexes, :])

# Does not work -> throws KeyError
a_df = df.loc[anomalies, :]
print(a_df)
print("-" * 10)
# print(a_df.__class__)
# print(a_df.head())
print(a_df.info())
print("-" * 10)
print(a_df.columns)
