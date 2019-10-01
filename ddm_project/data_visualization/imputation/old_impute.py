import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

df = pd.read_csv("air_passengers_v2.csv",
                 index_col="date", parse_dates=["date"])
# df.index.name = df.index.name.lower()
# df.columns = [c.lower() for c in df.columns]
# df.index = df.index.str.replace(r"(.*)-(.*)", r"\1-19\2")
# df.index = pd.to_datetime(df.index)

# df.to_csv("air_passengers_new.csv")
df["original"] = df.passengers.copy()
df.passengers[40:50] = np.nan

methods = ['linear', 'time', 'index', 'pad', 'nearest', 'zero', 'slinear',
           'quadratic', 'cubic', 'spline', 'barycentric', 'polynomial',
           'krogh', 'piecewise_polynomial', 'spline', 'pchip', 'akima',
           'from_derivatives']

for method in methods:
    if method in ['polynomial', 'spline']:
        interpolation = df.passengers.interpolate(method=method, order=2)
    else:
        interpolation = df.passengers.interpolate(method=method)
    df[method] = interpolation

plt.plot(df.original)
df.drop("original", axis=1)[40:50].plot(
    linestyle='--', ax=plt.gca(), alpha=0.3)


plt.show()
