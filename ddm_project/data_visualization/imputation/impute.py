import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

base_dir = "../../datasets/nab_dataset/data/"
# file_name = "realKnownCause/ambient_temperature_system_failure.csv"
# file_name = "realKnownCause/machine_temperature_system_failure.csv"
file_name = "realAWSCloudwatch/old_elb_request_count_8c0756.csv"

df = pd.read_csv(base_dir + file_name,
                 index_col="timestamp",
                 parse_dates=["timestamp"])

methods = ['linear', 'time', 'index', 'pad', 'nearest', 'zero', 'slinear',
           'quadratic', 'cubic', 'spline', 'barycentric', 'polynomial',
           'krogh', 'piecewise_polynomial', 'spline', 'pchip', 'akima',
           'from_derivatives']

print(df.index.has_duplicates)
print(df[df.index.duplicated()])

first_delta = df.index[1] - df.index[0]
wrong_deltas = []
for i in range(1, len(df)):
    delta = df.index[i] - df.index[i - 1]
    if delta != first_delta:
        wrong_deltas.append(i)
        print("{}: {}".format(i, delta))

# ax = df.plot()
fig, ax = plt.subplots()
ax.plot(df.index[wrong_deltas], df.iloc[wrong_deltas],
        "x", color="r", zorder=10, label="wrong deltas")
# plt.show()

# df = df.reindex(index=df.asfreq("5min").index)
# df.value = df.value.interpolate(method="linear")
# df.to_csv("interp_elb.csv", float_format="%.1f")
# raise SystemExit
old_df = df.copy()
df = df.reindex(index=df.asfreq("5min").index)
df['interpolated'] = df.value.interpolate(method="linear")
df.plot(ax=ax)
ax.lines = ax.lines[::-1]
plt.legend()
plt.show()
# raise SystemExit

# df = pd.read_csv("interp.csv", index_col="timestamp",
#                  parse_dates=["timestamp"])
# for method in methods:
#     if method in ['polynomial', 'spline']:
#         interpolation = df.value.interpolate(method=method, order=2)
#     else:
#         interpolation = df.value.interpolate(method=method)
#     df[method] = interpolation

# val_line = plt.plot(df.value)
# lines = plt.plot(df.iloc[:, 1:].where(
#     df.value.isna()), linestyle="--", alpha=0.4)
# plt.legend(val_line + lines, ["value"] + methods)
# ax = plt.gca()
# ax.plot(df.drop('value', axis=1), linestyle='-', alpha=0.3)
