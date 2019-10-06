"""Utilities functions."""

import matplotlib.pyplot as plt
import pandas as pd


def _format_parameters(d):
    """Format a dictionary of parameters to a string.

    Parameters
    ----------
    d : dict
        The dictionary to format.

    Returns
    -------
    str
        A string representation of input, useful as a key for dictionaries and
        as title in plots.

    Example: {"param1": 0.01, "param2": 67} -> "param1_0.01__param2_67"

    """
    finalstr = ""
    for k, v in d.items():
        if k == 'behaviour':
            continue
        if type(v) == str:
            pstr = str(k) + '_' + str(v)
        else:
            pstr = "{}_{:.4f}".format(k, v)
        finalstr = finalstr + pstr + "__"
    return finalstr.strip('_')


def _make_plots(
    df, tdf, labels, labels_windows, model_name, anomalies, params
):
    """Plot original data with found anomalies. Highlight anomaly windows.

    Parameters
    ----------
    model_name : str
        Name of the model used to compute the anomalies.
    anomalies : pd.Series
        Subset of the original data labeled as anomaly by the model.
    params : str
        Formatted string for parameters used in the model.

    Returns
    -------
    matplotlib.Axes
        The axis containing the plots.

    """
    fig, ax = plt.subplots()

    # Original data - dashed line
    ax.plot(
        df.index, df.value, label='Original data', linestyle='--', alpha=0.5
    )

    # Data effectively used
    ax.plot(tdf.index, tdf.value, label='Used data', color='C0')

    # Predicted anomalies plotted as 'x' points
    ax.plot(anomalies, 'x', label="Predicted anomalies")

    # Ground truth anomalies plotted as 'o' points
    ax.plot(labels, tdf.loc[labels], 'o',
            markersize=5, label="Real anomalies")

    y_min, y_max = ax.get_ylim()

    # Label windows as colored vertical intervals
    for win in labels_windows:
        ax.fill_between(win, y_min, y_max, color='r', alpha=0.1)
        # plt.plot(win, tdf.loc[win],
        #          '^', markersize=5, label="anomalies windows")
    ax.set_ylim(y_min, y_max)
    ax.set_title(model_name + " " + params)
    ax.legend()
    return ax


def get_gt_arrays(index, win_index, labels, labels_windows):
    """Construct ground truth arrays from labels and window labels."""
    gt_pred = pd.Series(1, index)
    gt_pred.loc[labels] = -1
    gt_windows = []
    idf = pd.DataFrame(index=win_index)
    idf['idx'] = idf.reset_index().index
    for win in labels_windows:
        win_start = idf.idx.at[win[0]]
        win_end = idf.idx.at[win[1]]
        gt_windows.append((win_start, win_end))
    return gt_pred, gt_windows
