import sys

import matplotlib.pyplot as plt
import numpy as np

from ddm_project.arima.evaluate_arima import get_arima_predictions

residuals = np.arange(20).astype(np.float64)
window_size = int(sys.argv[1])  # 50
alpha = float(sys.argv[2])  # 0.06
weights = (1 - alpha) ** np.arange(window_size)
weights = np.flip(weights)
print(weights)
weights = weights / weights.sum()
print(weights)
plt.plot(weights)
plt.show()

a = np.array([10, 2, 2])
a_w = a * weights
print("SMA: mean={:.3f} std={:.3f}".format(a.mean(), a.std()))
print(
    "EWMA: mean={:.3f} std={:.3f}".format(
        a_w.sum(), np.sqrt(np.dot(weights, (a - a_w.sum()) ** 2))
    )
)

print("Manually exiting...")
raise SystemExit

pred = get_arima_predictions(residuals, window_size, alpha)

# print(pred)
