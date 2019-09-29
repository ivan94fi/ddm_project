# noqa
import matplotlib.pyplot as plt
import numpy as np

A_TP = 1.
A_FP = 1.
# A_FN = 1.
# A_TN = 1.
C = 1.


def score(y):
    if y < -3:
        return -1.
    return 2 * (1 / (1 + np.exp(5 * y))) - 1


detected = [-5, -2, 0.45, 3]
scores = [score(y) for y in detected]
x = np.arange(-5, 3, 0.01)
plt.plot(x, [score(n) for n in x])
plt.plot(detected, scores, 'x')
print(detected)
print(scores)

S_FP = A_FP / 9

scores_weights = [(S_FP, scores[0]), (A_TP, scores[1]),
                  (S_FP, scores[2]), (S_FP, scores[3])]

for s in scores_weights:
    print("{:.3f} {:.3f}".format(s[0], s[1]))

print("")
print("Actual:", 0.11 * (-1 - 1 - 0.8093) + 0.9999)
# plt.show()

a = np.array(scores_weights)
s = a[:, 0] * a[:, 1]
print(s)
print(s.sum())
