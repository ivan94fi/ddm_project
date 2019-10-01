"""Plots for Trieste sea temperature dataset."""

import glob
import re

import matplotlib.pyplot as plt
import pandas as pd
from natsort import natsorted

from ddm_project.settings import datasets_dir

root_dir = datasets_dir + "/trieste_sea_temp_dataset"

data_paths = glob.glob(root_dir + "/*.csv")
data_paths = natsorted(data_paths)

# FIXME: does not work for all files. Must handle all possible columns names.
for i, p in enumerate(data_paths):
    # fig, ax = plt.subplots()
    print(p.split('/')[-1])
    if re.search("61490", p):
        df = pd.read_csv(p, skiprows=1, names=[
                         "DATE", "SEA TEMPERATURE", "FLAG"], index_col="DATE")
        df = df.drop("29/2/1900")
        df.index = pd.DatetimeIndex(df.index)
    elif re.search("61493", p):
        df = pd.read_csv(p, parse_dates=["YEAR"], index_col="YEAR")
    # print(df.head(10))
    else:
        df = pd.read_csv(p, parse_dates=["DATE"], index_col="DATE")
    df.plot()
    plt.show()
