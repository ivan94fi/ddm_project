"""This module contains the functions used to evaluate the AD models."""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

MAX_TP = 0.9866142981514305
TP_WEIGHT = 0.11


def get_simple_metrics(gt_pred, pred):
    """Compute precision, recall and F-score.

    Parameters
    ----------
    gt_pred : np.ndarray
        The ground truth anomalies. Anomalies are marked as -1, non anomalous
        observations are marked as 1.
    pred : np.ndarray
        The computed anomalies. Uses the same convention as `gt_pred`.

    Returns
    -------
    tuple
        The metrics as 3 float values: (precision, recall, F-score).

    """
    res = precision_recall_fscore_support(gt_pred, pred, beta=1.0, labels=[-1])
    return tuple(a.item() for a in res)[:3]


def get_nab_score(gt_windows, pred):
    """Compute NAB score.

    Compute a slight variation of the NAB score as descripted in 'Evaluating
    Real-time Anomaly Detection Algorithms - the Numenta Anomaly Benchmark' by
    Alexander Lavin and Subutai Ahmad.

    Parameters
    ----------
    gt_windows : list of tuples
        List of windows start / end positions.
    pred : list
        The predicted anomaly labels.

    Returns
    -------
    type
        The summed scores for the predicted anomalies.

    """
    scores = []
    windows_number = len(gt_windows)

    # Handle anomalies before first window -> False positives
    early_anomalies_score = (pred[:gt_windows[0][0]] < 0).sum()

    # Handle false negatives (window without detection)
    windows_without_detections = [
        (pred[w[0]:w[1]] != -1).all() for w in gt_windows]
    false_negatives_score = sum(windows_without_detections)

    for i, window in enumerate(gt_windows):
        last_window = True if i == windows_number - 1 else False

        window_left = window[0]
        window_right = window[1]
        window_size = window_right - window_left
        scope_start = window_left
        scope_end = gt_windows[i + 1][0] - \
            1 if not last_window else pred.size

        # In window
        for j in range(scope_start, scope_end):
            p = pred[j]
            if p == 1:
                continue
            rel_pos = get_relative_position(j, window_right, window_size)
            score = get_raw_score(rel_pos) * TP_WEIGHT
            scores.append(score)

    anomalies_score = sum(scores)

    return anomalies_score - early_anomalies_score - false_negatives_score


def sigmoid(x):
    """Compute a sigmod function.

    Parameters
    ----------
    x : float
        The point where to evaluate the sigmoid.

    Returns
    -------
    float
        The sigmoid value in x.

    """
    return 1 / (1 + np.exp(-x))


def scaled_sigmoid(y):
    """Scale the sigmoid according to the definition in the original paper.

    Parameters
    ----------
    y : float
        Relative position inside the window.

    Returns
    -------
    float
        The scaled sigmoid value.

    """
    # Very far from right edge of the window. Simply return -1: the point is
    # considered a false negative.
    if y > 3:
        return -1.
    return 2 * sigmoid(-5 * y) - 1


def get_raw_score(y):
    """Get the unnormalized NAB score.

    Parameters
    ----------
    y : float
        Relative position inside the window.

    Returns
    -------
    float
        The unnormalized NAB score.

    """
    return np.clip(scaled_sigmoid(y) / MAX_TP, -1, 1)


def get_relative_position(pos, win_right, win_size):
    """Transform the absolute position of the anomaly in a relative position.

    Parameters
    ----------
    pos : int
        The absolute position of the anomaly. It is the index of the anomaly in
        the data.
    win_right : int
        The right edge of the window.
    win_size : int
        The size of the current window.

    Returns
    -------
    float
        The relative position of the anomaly at index `pos`.

    """
    return - (win_right - pos) / win_size


if __name__ == '__main__':
    from ddm_project.readers.nab_dataset_reader import NABReader

    np.random.seed(42)
    reader = NABReader()
    reader.load_data()
    reader.load_labels()

    dataset_name = "iio_us-east-1_i-a2eb1cd9_NetworkIn.csv"

    df = reader.data.get(dataset_name)
    labels = reader.labels.get(dataset_name)
    labels_windows = reader.label_windows.get(dataset_name)

    gt_windows = []
    idf = pd.DataFrame(index=df.index)
    idf['idx'] = idf.reset_index().index

    for win in labels_windows:
        win_start = idf.idx.at[win[0]]
        win_end = idf.idx.at[win[1]]
        gt_windows.append((win_start, win_end))

    pred = np.full((len(df),), 1)
    idx = [np.random.rand() < 0.1 for i in range(len(df))]
    pred[idx] = -1

    anomalies = df[pred == -1]
    print("anomalies percentage: {:.6f}".format(anomalies.shape[0] / len(df)))

    final_score = get_nab_score(gt_windows, pred)
    print("final score:", final_score)
