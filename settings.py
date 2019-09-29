"""
settings.py.

Define global variables.
"""
import os

data_root_dir = ".."
datasets_dir = os.path.join(data_root_dir, "datasets")

# NAB dataset
nab_root_dir = os.path.join(datasets_dir, "nab_dataset")
nab_data_dir = os.path.join(nab_root_dir, "data")
nab_label_dir = os.path.join(nab_root_dir, "labels")
nab_label_filename = "combined_labels.json"
nab_label_windows_filename = "combined_windows.json"
nab_label_path = os.path.join(nab_label_dir, nab_label_filename)
nab_label_windows_path = os.path.join(
    nab_label_dir, nab_label_windows_filename)
