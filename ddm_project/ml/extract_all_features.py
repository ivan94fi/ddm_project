"""
Script to iterate the chosen datasets from NAb and extract their features.

The features are saved in the `metrics` directory as pkl files.
"""

import logging

from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from ddm_project.ml.feature_generation import FeatureGenerator
from ddm_project.readers.nab_dataset_reader import NABReader

reader = NABReader()
reader.load_data()
reader.load_labels()

# Disable warnings for tsresh features extraction.
logging.getLogger(
    "tsfresh.utilities.dataframe_functions").setLevel(logging.ERROR)
logging.getLogger(
    "tsfresh.feature_selection.significance_tests").setLevel(logging.ERROR)
logging.getLogger(
    "tsfresh.feature_selection.relevance").setLevel(logging.ERROR)


dataset_names = ["iio_us-east-1_i-a2eb1cd9_NetworkIn.csv",
                 "machine_temperature_system_failure.csv",
                 "ec2_cpu_utilization_fe7f93.csv",
                 "rds_cpu_utilization_e47b3b.csv",
                 "grok_asg_anomaly.csv",
                 "elb_request_count_8c0756.csv"]


for name in tqdm(dataset_names):
    tqdm.write("Processing dataset {}".format(name))
    df = reader.data.get(name)
    scaler = StandardScaler()
    feature_generator = FeatureGenerator(
        df, ts_col='value', scaler=scaler,
        fname=name.replace(".csv", ".pkl")
    )
    X = feature_generator.get(read=False)
