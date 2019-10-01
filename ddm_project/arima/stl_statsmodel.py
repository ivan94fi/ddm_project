"""Trying STL from statsmodels."""

import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from statsmodels.datasets import co2
from statsmodels.tsa.seasonal import STL

register_matplotlib_converters()
data = co2.load(True).data

"""
- Misure sesonality dopo decomposizione:
    - F_T = max {0, 1 - Var(R_t)/(Var(T_t + R_t)) }
    - F_S = max {0, 1 - Var(R_t)/(Var(S_t + R_t)) }
"""


data = data.resample('M').mean().ffill()
res = STL(data).fit()


def get_seasonality_strength():
    """Return seasonality strength in [0,1], as defined in Rob Hyndman book."""
    residual_std = res.resid.std()
    strength = residual_std / (res.seasonal.std() + residual_std)
    return max(0, 1 - strength) ** 2


def get_trend_strength():
    """Return trend strength in [0,1], as defined in Rob Hyndman book."""
    residual_std = res.resid.std()
    strength = residual_std / (res.trend.std() + residual_std)
    return max(0, 1 - strength) ** 2


print("Seasonality strength:", get_seasonality_strength())
print("Trend strength:", get_trend_strength())

res.plot()
plt.show()
