import glob

import matplotlib.pyplot as plt
import pandas as pd
from natsort import natsorted

root_dir = "nasa_valve_dataset/Waveform Data/COL 1 Time COL 2 Current"

# print(glob.glob(root_dir + "/*.CSV"))

filepaths = natsorted(glob.glob(root_dir + "/*.CSV"))

blacklist = ['TEK0000{}.CSV'.format(i) for i in range(4, 10)]

filepaths = [f for f in filepaths if f.split('/')[-1] not in blacklist]


fig, axes = plt.subplots(2, 6)

for i, p in enumerate(filepaths):
    df = pd.read_csv(p,
                     header=None,
                     names=('a', 'b'))
    print(len(df))
    color = 'green' if i < 4 else 'red'
    ax = axes[i // 6, i % 6]
    ax.plot(df.iloc[:, 0], df.iloc[:, 1], color=color)
    ax.set_title(p.split('/')[-1])
plt.show()
