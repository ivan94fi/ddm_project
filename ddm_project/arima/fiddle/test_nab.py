"""
This file was originally used to plot the datasets from NAB.

Now it also contains some statistical plots (autocorrelation plot,
partial autocorrelation plot, lag plots)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pandas.plotting import lag_plot  # , autocorrelation_plot

from ddm_project.settings import datasets_dir

root_dir = datasets_dir + "/nab_dataset/data/"

dataset = "realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv"

# df = pd.read_csv(root_dir + dataset,
#                  parse_dates=['timestamp'], index_col='timestamp')

df = pd.DataFrame({"value": np.random.randn(100)})

# plot_acf(df['value'], lags=range(len(df)), zero=False,
#   title="statsmodel",fft=True)
# plt.figure()
# autocorrelation_plot(df.value.tolist())
# plt.show()
# plt.gca().set_xlim(right=500)
lags = 20  # 500
fig, axes = plt.subplots(2, 1, sharex=True)
sm.graphics.tsa.plot_acf(df['value'], lags=lags, ax=axes[0], marker=None)
sm.graphics.tsa.plot_pacf(df['value'], lags=lags, ax=axes[1], marker=None)
# sm.graphics.tsa.plot_acf(df['value'], lags=lags, marker=None)
# sm.graphics.tsa.plot_pacf(df['value'], lags=lags, marker=None)
plt.show()
raise SystemExit

fig, axes = plt.subplots(2, 5)
axes = axes.ravel()
freq_range = range(285, 295)
for i, ax in enumerate(axes):
    lag_plot(df['value'], lag=freq_range[i], ax=ax)
    ax.set_title("lag {}".format(freq_range[i]))


# freq_range = range(285,295)
# fig, axes = plt.subplots(len(freq_range), 3, sharex=True, figsize=(10, 12))
#
# for i, freq in enumerate(freq_range):
#    mul_dec = seasonal_decompose(df['value'], model='mul', freq=freq)
#    axes[i,0].plot(mul_dec.trend)
#    axes[i,1].plot(mul_dec.seasonal)
#    axes[i,2].plot(mul_dec.resid)
#    # axes[i].set_title(str(freq))

plt.show()
