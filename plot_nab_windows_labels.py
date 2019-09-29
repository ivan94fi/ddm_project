# noqa
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

from nab_dataset_reader import NABReader

register_matplotlib_converters()

reader = NABReader()
reader.load_data()
reader.load_labels()

dataset_name = "iio_us-east-1_i-a2eb1cd9_NetworkIn.csv"
df = reader.data.get(dataset_name)
labels = reader.labels.get(dataset_name)
labels_windows = reader.label_windows.get(dataset_name)
print(labels_windows)
raise SystemExit
# print(dataset_name, labels_windows[1] - labels_windows[0])

# Print size of windows
# for key in reader.labels._dict.keys():
#     df = reader.data.get(key)
#     if df is None:
#         continue
#     labels_windows = reader.label_windows.get(key)
#     string = key + ":"
#     for win in labels_windows:
#         string = string + " " + str(len(df.loc[win[0]:win[1]]))
#     print(string)
#     print("")
# raise Exception

for key in reader.labels._dict.keys():
    labels = reader.labels.get(key)
    labels_windows = reader.label_windows.get(key)
    if not labels or not labels_windows:
        continue
    for i in range(len(labels)):
        print(key)
        line, = plt.plot(labels[i], 1, 'x', label='anomaly')
        try:
            plt.vlines(labels_windows[i], 0, 2,
                       label="windows", color=line.get_color())
        except IndexError:
            plt.text(labels[i], 1.1, "Window not defined",
                     color=line.get_color())
    plt.title(key)
    plt.legend()
    plt.show()
