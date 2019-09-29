import logging

from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from features_generation import FeatureGenerator
from nab_dataset_reader import NABReader

reader = NABReader()
reader.load_data()
reader.load_labels()


def _get_loggers():
    return [logging.getLogger(name)
            for name in logging.root.manager.loggerDict]


tsfresh_df_functions_logger = logging.getLogger(
    "tsfresh.utilities.dataframe_functions")
tsfresh_signifiance_tests_logger = logging.getLogger(
    "tsfresh.feature_selection.significance_tests"
)
tsfresh_relevance_logger = logging.getLogger(
    "tsfresh.feature_selection.relevance"
)
tsfresh_df_functions_logger.setLevel(logging.ERROR)
tsfresh_signifiance_tests_logger.setLevel(logging.ERROR)
tsfresh_relevance_logger.setLevel(logging.ERROR)


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
