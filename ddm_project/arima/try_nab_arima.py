"""
First attempt to define the anomaly detection pipeline with ARIMA.

Capire stationarity:
- Time plot -> capire se c'è trend/seasonality/necessità log transform
- ACF, PACF e Box-Ljung test -> da qui si vede se c'è autocorrelazione
- Test statistici per capire stazionarità:
    - KPSS: Applicazione ripetuta dà numero differenziazioni che sono
        da applicare.
    - ADF: Augmented Dickey-Fuller unit root test
- Capire seasonality:
    - ACF
    - Seasonality decomposition
    - Misure sesonality dopo decomposizione:
        - F_T = max {0, 1 - Var(R_t)/(Var(T_t + R_t)) }
        - F_S = max {0, 1 - Var(R_t)/(Var(S_t + R_t)) }

Differenziazione per stazionarizzare
- Dopo eventuale log, fare sesonal differencing se necessario e poi
  differencing normale

Capire valori p e q per AR e MA:
- Usare ACF-PACF (solo se dati vengono da ARIMA(p,d,0) o ARIMA(0,d,q))
- Usare AIC_c per scegliere modello migliore
- Controllare residuals (ACF, portmanteau)
- Calcolare forecast quando residuals sono soddisfacendti
"""

import warnings
from collections import namedtuple

import matplotlib.pyplot as plt
import statsmodels.tsa.api as tsa
from pandas.plotting import register_matplotlib_converters
from pmdarima.arima import ARIMA, auto_arima

from ddm_project.readers.nab_dataset_reader import NABReader

register_matplotlib_converters()

reader = NABReader()
reader.load_data()
reader.load_labels()

df = reader.data.get("iio_us-east-1_i-a2eb1cd9_NetworkIn.csv")

# df.plot()
# plt.show()

# Train-test split
train_percentage = 0.8
train_len = int(train_percentage * len(df))
data = df.value.values  # / df.value.values.max()
train, test = data[:train_len], data[train_len:]

# KPSS test
KPSSResults = namedtuple(
    "KPSSResults", ["kpss_stat", "p_value", "lags", "critical_values"])
kpss_results = KPSSResults(*tsa.kpss(data, nlags='auto'))
print("KPSS results:\n", kpss_results)

auto_fit = False
if auto_fit:
    arima = auto_arima(train, stepwise=True, trace=1, seasonal=False)
    print(arima.summary())
else:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arima = ARIMA(order=(4, 1, 4), seasonal_order=None)
        arima.fit(train)

# Diagnostics plot
arima.plot_diagnostics(lags=50)
plt.gcf().suptitle('Diagnostics Plot', fontsize=14)

# !! not necessary !! Everything already plotted
# Plot Residuals and fitted values
# plt.figure()
# fitted_values = arima.predict_in_sample()
# plt.plot(df.index[:train_len - 1], fitted_values,
#          color='C0', label="Fitted values")
# plt.plot(pd.to_datetime(df.index), data, color='C1', label="Data")
# plt.plot(df.index[:train_len - 1], arima.resid(),
#          color='C2', label="Residuals")
# plt.gca().grid(which='both', axis='x', linestyle='--')
# plt.title("Residuals and fitted values")
# plt.legend()

print("SSE: {}".format((arima.resid() ** 2).sum()))

# Plot fitted values and forecasts
predictions = arima.predict(n_periods=test.shape[0])
fitted_values = arima.predict_in_sample()
plt.figure()
plt.plot(df.index[train_len:], test, '--', color='C0', label="test set")
plt.plot(df.index[train_len:], predictions,
         '--', color='C1', label="forecasted values")
plt.plot(df.index[:train_len], train, color='C0', label="train set")
plt.plot(df.index[:train_len - 1], fitted_values,
         color='C1', label="fitted values")
plt.legend()
plt.title("Fitted values and forecasts")

plt.show()
