import glob
import json
import re
from collections import defaultdict
from pprint import pprint  # noqa

import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

"""
realAWSCloudwatch/ec2_cpu_utilization_fe7f93.csv -> ok
realAWSCloudwatch/rds_cpu_utilization_e47b3b.csv -> ok
realAWSCloudwatch/grok_asg_anomaly.csv -> ok
realAWSCloudwatch/iio_us-east-1_i-a2eb1cd9_NetworkIn.csv -> ok

realKnownCause/machine_temperature_system_failure.csv -> file corretto -> ok
realAWSCloudwatch/elb_request_count_8c0756.csv linear interpolation -> ok

realKnownCause/ambient_temperature_system_failure.csv ->
    10 delta diversi -> necessaria correzione stagionale con STL
"""


root_dir = "../../datasets/nab_dataset/data"
label_dir = "../../datasets/nab_dataset/labels"
label_file = "combined_labels.json"

label_path = label_dir + "/" + label_file
with open(label_path) as f:
    labels = json.load(f)

data_paths = glob.glob(root_dir + "/realAWSCloudwatch/*.csv")
data_paths += glob.glob(root_dir + "/realKnownCause/*.csv")
pattern = re.compile(r'\/([\d\w_-]*)\.')
data_filenames = [re.search(pattern, p).group(1) for p in data_paths]

# ===========================
_fname = data_paths[15]
_name = _fname.split('/', 5)[-1]
anomalies = labels[_name]
df = pd.read_csv(_fname, parse_dates=['timestamp'], index_col='timestamp')
values = df.loc[[pd.to_datetime(a) for a in anomalies]]
plt.plot(df.index.to_pydatetime(), df['value'])
plt.scatter(anomalies, values, marker="x", color="orange", s=80, zorder=10)

plt.title(_name)
plt.show()
raise SystemExit
# ===========================


plot = True
if plot:
    fig, axes = plt.subplots(4, 6)

# df = pd.read_csv(data_paths[0], parse_dates=[
#                  'timestamp'], index_col='timestamp')
# first_delta = df.index[1] - df.index[0]
# print("first delta:", first_delta)
#
# raise SystemExit
all_deltas = {}
equally_spaced = []
non_equally_spaced = []
for i, path in enumerate(data_paths):
    name = path.split('/', 5)[-1]
    df = pd.read_csv(path, parse_dates=['timestamp'], index_col='timestamp')
    deltas = defaultdict(lambda: 0)
    first_delta = df.index[1] - df.index[0]
    print(path)
    print("first delta:", first_delta)

    for j in range(1, len(df)):
        delta = df.index[j] - df.index[j - 1]
        deltas[delta] += 1
        if delta != first_delta:
            print("{}: {}".format(j, delta))
    all_deltas[name] = deltas
    if len(deltas) == 1:
        equally_spaced.append(name)
    else:
        non_equally_spaced.append(name)
    print("")

    max_count = 0
    max_delta = None
    total_counts = 0
    for delta, count in deltas.items():
        if count > max_count:
            max_count = count
            max_delta = delta
        total_counts += count

    # total = sum([count for _, count in deltas.items()])
    assert total_counts == len(df) - 1,\
        "Expected {} to be {}.".format(total_counts, len(df) - 1)
    if plot:
        ax = axes[i // 6, i % 6]
        # fig, ax = plt.subplots()
        ax.plot(df.index.to_pydatetime(), df['value'])
        anomalies = labels[name]
        values = df.loc[[pd.to_datetime(a) for a in anomalies]]
        ax.scatter(anomalies, values, marker="x",
                   color="orange", s=80, zorder=10)
        if len(deltas) != 1:
            ax.set_facecolor((235 / 255, 64 / 255, 52 / 255, 0.1))
            y_min, y_max = ax.get_ylim()
            print("min max:", y_min, y_max)
            for j in range(1, len(df)):
                delta = df.index[j] - df.index[j - 1]
                if delta != max_delta:
                    # print(delta)
                    # print(df.index[j - 1], df.index[j])
                    # print(df.index[j - 1:j + 1])
                    # raise SystemExit
                    ax.fill_between(
                        df.index[j - 1:j + 1],
                        y_min, y_max,
                        color='r', alpha=0.5)
            ax.set_ylim(y_min, y_max)

        ax.set_title(name.replace('/', ' \n '), fontsize=7)
        ax.xaxis.set_ticklabels([])
        ax.xaxis.set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.tick_params(axis='both', which='minor', labelsize=6)
        ax.yaxis.get_offset_text().set_fontsize(6)

if plot:
    # plt.show()
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    fig.set_tight_layout(True)
    plt.show()
    # plt.savefig("nab_dataset_plots.svg", transparent=False, dpi=100,
    #             frameon=False, bbox_inches='tight', pad_inches=0)

print("")
print("=" * 80)
print("")
for name, delta in all_deltas.items():
    print(name)
    for k, v in delta.items():
        print("{}: {}".format(k, v))

print("")
print("=" * 80)
print("")

for name, delta in all_deltas.items():
    print("{}: {}".format(name, len(delta)))
# '2014-04-11 00:00:00'
# ax.plot([datetime.datetime('2014-04-10 07:15:00.000000'),
#   datetime.datetime('2014-04-11 16:45:00.000000')], [50, 50], color='orange')

# ax.plot([datetime.datetime(year=2014, month=4, day=10, hour=7, minute=15),
#   datetime.datetime(year=2014, month=4, day=11, hour=16, minute=45)],
#   [50, 50], color='orange')
# ax.scatter([datetime.datetime(year=2014, month=4, day=11)], [60],
#   color='red')

# plt.show()
