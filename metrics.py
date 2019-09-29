import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from nab_dataset_reader import NABReader

MAX_TP = 0.9866142981514305
TP_WEIGHT = 0.11

"""
Sostanzialmente non si fa mai lo score di una relative position < -1 perché
si contano solo punti dentro window. Se siamo fuori da una window, si usa la
funzione della window precedente.
L'if con np.abs in scaled_sigmoid non importa (basta > 1, tanto si checkano
solo punti in window.)
"""

"""
Algoritmo per calcolo score:

for each point o labeled as outlier by the model
    if o è in una finestra
        win = finestra di o
        score = score con peso appropriato
    else o è fuori da una finestra:
        win = finestra precedente ad o
        if win non esiste
            score = -1
        else score con peso appropriato
    salva score in una lista_score
score finale = somma lista_score + peso falsi negativi
"""


def get_simple_metrics(gt_pred, pred):
    # Return precision, recall, F-score.
    res = precision_recall_fscore_support(gt_pred, pred, beta=1.0, labels=[-1])
    return tuple(a.item() for a in res)[:3]


def get_nab_score(gt_windows, pred):
    """Short summary.

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
    # print("early:", early_anomalies_score)

    # Handle false negatives (window without detection)
    windows_without_detections = [
        (pred[w[0]:w[1]] != -1).all() for w in gt_windows]
    false_negatives_score = sum(windows_without_detections)
    # print("fn score:", false_negatives_score)

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
    # print("anomalies:", anomalies_score)
    return anomalies_score - early_anomalies_score - false_negatives_score


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def scaled_sigmoid(y):
    if y > 3:
        return -1.
    return 2 * sigmoid(-5 * y) - 1


def get_raw_score(y):
    return np.clip(scaled_sigmoid(y) / MAX_TP, -1, 1)


def get_relative_position(pos, win_right, win_size):
    scaled_pos = - (win_right - pos) / win_size
    return scaled_pos


if __name__ == '__main__':
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
