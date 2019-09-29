import glob
import re
from pprint import pprint  # noqa

import matplotlib.pyplot as plt
import pandas as pd
from natsort import natsorted

# import numpy as np
# import statsmodels.api as sm
# from pandas.plotting import autocorrelation_plot, lag_plot
# from statsmodels.tsa.seasonal import seasonal_decompose

"""
1,8,9,13,18*,23*,24*,27*,29*,38*,54*,65
"""

chosen = [1, 8, 9, 13, 18, 23, 24, 27, 29, 38, 54, 65]

root_dir = "../../datasets/yahoo_ad_dataset/"
root_dir = root_dir + "ydata-labeled-time-series-anomalies-v1_0/A1Benchmark"

data_paths = glob.glob(root_dir + "/*.csv")
data_paths = natsorted(data_paths)
data_paths = [data_paths[i - 1] for i in chosen]
pattern = re.compile(r'\/([\d\w_-]*\.csv)$')
data_filenames = [re.search(pattern, p).group(1) for p in data_paths]

# print(data_paths)
# print(data_filenames)


df = pd.read_csv(data_paths[0], index_col="timestamp")

# raise SystemExit
for i, p in enumerate(data_paths):
    fig, ax = plt.subplots()
    df = pd.read_csv(p, index_col="timestamp")
    # df[df['is_anomaly']]
    df['anomaly'] = df['value'].mask(df['is_anomaly'] == 0)
    ax.plot(df['value'])
    # ax.plot(df['is_anomaly'])
    ax.scatter(range(len(df)), df['anomaly'],
               marker="x", color="orange", s=80, zorder=10)
    name = p.split('/')[-1]
    ax.set_title(name)
    plt.savefig("yahoo_df_and_anomalies_plots/plot_" + name + ".svg")
    plt.close(fig)
    plt.show()
