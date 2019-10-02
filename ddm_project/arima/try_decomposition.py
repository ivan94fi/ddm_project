"""Trying decomposition functions."""

import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from statsmodels.datasets import co2

from ddm_project.arima.decomposition import (
    decompose, get_seasonality_strength, get_trend_strength
)

register_matplotlib_converters()
data = co2.load(True).data

n = len(data)
half = int(n / 2)
# data.co2.iloc[half:half + 10] += 50

data = data.resample('M').mean().ffill()
ma_decomposition = decompose(data, method="MA")
decomposition = decompose(data)
robust_decomposition = decompose(data, robust=True)

print("Seasonality strength:", get_seasonality_strength(data))
print("Trend strength:", get_trend_strength(data))

fig = decomposition.plot()
fig.suptitle('Non-robust STL')
ma_fig = ma_decomposition.plot()
ma_fig.suptitle('MA')
robust_fig = robust_decomposition.plot()
robust_fig.suptitle('Robust STL')
plt.show()
