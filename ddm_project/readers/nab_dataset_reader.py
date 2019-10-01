"""
nab_dataset_reader.py.

realKnownCause/machine_temperature_system_failure.csv
realAWSCloudwatch/ec2_cpu_utilization_fe7f93.csv
realKnownCause/ambient_temperature_system_failure.csv -> missing values
realAWSCloudwatch/iio_us-east-1_i-a2eb1cd9_NetworkIn.csv
"""
import collections
import glob
import json
import re

import pandas as pd

from ddm_project.settings import (
    nab_data_dir, nab_label_path, nab_label_windows_path
)


class LabelsDict(object):
    """Dict that converts to datetime on access."""

    def __init__(self, _dict):
        self._dict = _dict

    def __len__(self):
        return len(self._dict)

    def __iter__(self):
        return iter(self._dict)

    def __getitem__(self, key):
        val = self._dict[key]
        if isinstance(val, collections.Iterable):
            return [pd.to_datetime(d) for d in val]
        else:
            return pd.to_datetime(val)

    def __setitem__(self, key, value):
        self._dict[key] = value

    def get(self, key, value=None):
        try:
            return self.__getitem__(key)
        except KeyError:
            return value


class NABReader(object):
    """
    Class for reading dataframes in NAB dataset.

    Args: paths (list): the paths to the dataframes to load. if None read the
     default data.
    """

    fname_pattern = re.compile(r'\/([\d\w_-]*\.csv)')

    def __init__(self, paths=None):

        if not paths:
            paths = glob.glob(nab_data_dir + "/realAWSCloudwatch/*.csv") \
                + glob.glob(nab_data_dir + "/realKnownCause/*.csv")

        self.paths = paths
        self.data = None
        self.labels = None
        self.label_windows = None

    def path_to_fname(self, p):
        return re.search(self.fname_pattern, p).group(1)

    def load_data(self):
        """
        Load the dataframes specified in self.paths in a dictionary with
        filenames as keys.
        """
        date_col = 'timestamp'
        kwargs = {
            'parse_dates': [date_col],
            'index_col': date_col
        }
        self.data = {self.path_to_fname(p): pd.read_csv(p, **kwargs)
                     for p in self.paths}

    def load_labels(self, transform=lambda k: k.split('/')[1]):
        """
        Load the labels json file in a dictionary. Use the transformation to
        modify the keys.
        """
        with open(nab_label_path) as f:
            labels = json.load(f)
        self.labels = LabelsDict({transform(k): v for k, v in labels.items()})

        with open(nab_label_windows_path) as f:
            labels_windows = json.load(f)
        self.label_windows = LabelsDict(
            {transform(k): v for k, v in labels_windows.items()})


if __name__ == '__main__':

    reader = NABReader()
    reader.load_data()
    reader.load_labels()

    # print(reader.labels)
    # print(reader.data)
    # print(reader.label_windows)

    dataset_name = "iio_us-east-1_i-a2eb1cd9_NetworkIn.csv"
    win = reader.label_windows.get(dataset_name)
    print(win)

    # l = [v.memory_usage()['value'] * 2 for k, v in a.df_dict.items()]
    # fig, axes = plt.subplots(4, 6)
    #
    # for i, path in enumerate(data_paths):
    #     ax = axes[i // 6, i % 6]
    #     df = pd.read_csv(path,
    #                      parse_dates=['timestamp'],
    #                      index_col='timestamp')
    #     # fig, ax = plt.subplots()
    #     ax.plot(df.index.to_pydatetime(), df['value'])
    #     name = re.search(pattern, path).group(1)
    #     print(name)
    #     print(path.split('/', 2))
    #     anomalies = labels[name]
    #     values = df.loc[[pd.to_datetime(a) for a in anomalies]]
    #     ax.scatter(anomalies, values, marker="x", color="orange", s=80)
    #     ax.set_title(name.replace('/', ' \n '), fontsize=7)
    #     ax.xaxis.set_ticklabels([])
    #     ax.xaxis.set_visible(False)
    # plt.show()

# '2014-04-11 00:00:00'
# ax.plot([datetime.datetime('2014-04-10 07:15:00.000000'),
#   datetime.datetime('2014-04-11 16:45:00.000000')], [50, 50], color='orange')

# ax.plot([datetime.datetime(year=2014, month=4, day=10, hour=7, minute=15),
#   datetime.datetime(year=2014, month=4, day=11, hour=16, minute=45)],
#   [50, 50], color='orange')
# ax.scatter([datetime.datetime(year=2014, month=4, day=11)],
# [60], color='red')

# plt.show()
