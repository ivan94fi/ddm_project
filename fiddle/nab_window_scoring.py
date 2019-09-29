import matplotlib.pyplot as plt
import numpy as np

max_tp = 0.9866142981514305


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def scaled_sigmoid(y):
    if np.abs(y) > 1:
        return -1.
    return 2 * sigmoid(-5 * y) - 1


def get_score(x):
    return np.clip(scaled_sigmoid(x) / max_tp, -1, 1)


def get_relative_position(pos, win_right, win_size):
    scaled_pos = - (win_right - pos) / win_size
    return scaled_pos


if __name__ == '__main__':

    win_left = 3
    win_right = 12
    win_size = win_right - win_left
    absolute_pos = [2, 3, 4, 5, 7, 8, 11, 12, 13]

    tp_weight = 0.11
    relative_pos = [get_relative_position(
        x, win_right, win_size) for x in absolute_pos]

    scores = [get_score(x) for x in relative_pos]

    print(scores)

    rng = np.arange(-4, 4, 0.01)
    plt.plot(rng, [get_score(x) for x in rng])
    plt.plot(rng, [scaled_sigmoid(x) for x in rng])
    plt.show()
    raise SystemExit("--  Manually interrupted  --")

    raw_scores = [scaled_sigmoid(x) for x in relative_pos]
    scaled_scores = [x / max_tp for x in raw_scores]
    scores = [x * tp_weight for x in scaled_scores]
    print("abs_pos\trel_pos\traw\tscaled\tscore")
    for abs_pos, rel_pos, raw, scaled, score in zip(
            absolute_pos, relative_pos, raw_scores, scaled_scores, scores):
        print("{:2d}\t{: .3f}:\t{: .3f}\t{: .3f}\t{: .3f}".format(
            abs_pos, rel_pos, raw, scaled, score))

    # unweighted = [nab_scaledSigmoid(i) for i in rel_positions]
    # plt.plot(rel_positions, unweighted, label='nab unweighted')
    # weighted = [s * 1 / max_tp for s in unweighted]
    # plt.plot(rel_positions, weighted, label='nab weighted')
    # ######
    # plt.plot(rel_positions, [scaled_sigmoid(i)
    #                          for i in rel_positions], label="mia")
    # plt.legend()
    # plt.show()
