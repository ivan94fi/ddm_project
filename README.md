# DDM Project

Project for Data and Document Mining exam.

## Modules

This python package encloses the scripts used to generate results for the project.

The main module is `ddm_project` and the structure of the submodules is the following:

`ddm_project`</br>
&nbsp;&nbsp;&nbsp;├── `arima`: scripts to create results for ARIMA models.</br>
&nbsp;&nbsp;&nbsp;├── `data_visualization`: scripts to visualize the datasets.</br>
&nbsp;&nbsp;&nbsp;├── `metrics`: functions for metris calculation.</br>
&nbsp;&nbsp;&nbsp;├── `ml`: scripts to create results for machine learning models.</br>
&nbsp;&nbsp;&nbsp;├── `readers`: classes for data and labels reading.</br>
&nbsp;&nbsp;&nbsp;├── `settings.py`: file containing global variable definitions.</br>
&nbsp;&nbsp;&nbsp;└── `utils`: utilities functions.</br>

## Installation

The package requires Python 3.7 to run. To install the package it is necessary to change the path of the datasets folder in the `settings.py` file: just change the string assinged to the variable `data_root_dir`.

Once the variable points to a valid path, that is to the parent directory of `datasets`, it is possible to install the package by running the following command:
```shell
# First, ensure that you are in the root of the repository, the directory containing setup.py
python -m pip install .
```

If the installation succeeds you will be able to import `ddm_project` from anywhere in your system:
```
python -c "import ddm_project"
```

## Usage
### ARIMA procedures
To get ARIMA results use the script `evaluate_arima.py` in `ddm_project/arima`:
```shell
cd ddm_project/arima
python evaluate_arima.py N
```
where `N` is a number in `[0,5]` representing the dataset index.

The scripts utilizes the parameters found during the project, which are saved in `ddm_project/arima/saved_arima_params.yml`. The procedure to decide parameters for arima models is to execute the script `ddm_project/arima/find_arima_order.py` with the appropriate parameters. This procedure is not documented here.

### Machine Learining procedures
To use machine learning models it is possible to extract features for all datasets in advance. To do this use `ddm_project/ml/extract_all_features.py`:
```shell
cd ddm_project/ml
python extract_all_features.py
```
This script takes a while to complete but constantly informs you of the remaining time. The features will be then saved on disk and reused in the following scripts.

Otherwise, `grid_search.py` is capable of extracting features for the selected datast only, if they are not found on disk. To perform grid search on the models specify a dataset index as in the previous examples. The following is an example execution:
```shell
python grid_search.py 0 --iforest --ocsvm --plot_metrics
```
The above execution selects the dataset with index `0`, includes both the Isolation Forest model and the One-Class SVM model, and asks the script to plot how the metrics change with the various grid search parameters. It is possible to use the `--plot_predictions` switch to also plot the predicted anomalies on the data, together with the ground truth anomalies and anomaly windows.


