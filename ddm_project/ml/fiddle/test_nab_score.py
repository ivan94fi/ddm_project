"""This file `tests` the expected behaviour of the nab scoring metric."""

import numpy as np

from ddm_project.metrics.metrics import TP_WEIGHT, get_nab_score, get_raw_score

if __name__ == '__main__':
    gt_windows = [(3, 6), (9, 13), (24, 29)]
    fn_num = 1
    pred_num = 30
    pred = np.ones(pred_num)
    anomalies_positions = [0, 2, 4, 6, 7, 9, 11, 13, 16, 17, 19]
    early_anomalies = 2
    anomalies_num = len(anomalies_positions)
    pred[anomalies_positions] = -1

    rel_positions = [-2 / 3, 0, 1 / 3, -1, -1 / 2, 0, 3 / 4, 1, 6 / 4]
    expected_anomalies_score = np.sum(
        [get_raw_score(y) * TP_WEIGHT for y in rel_positions]
    )

    expected_score = expected_anomalies_score - early_anomalies - fn_num

    actual_score = get_nab_score(gt_windows, pred)

    assert actual_score == expected_score,\
        "actual={:.4f} expected={:.4f}".format(actual_score, expected_score)
    print("Passed")
