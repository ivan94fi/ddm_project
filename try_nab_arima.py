import sys

import matplotlib.pyplot as plt
import statsmodels.tsa.api as tsa
from pmdarima.arima import ARIMA, auto_arima

from nab_dataset_reader import NABReader

reader = NABReader()
reader.load_data()
reader.load_labels()

df = reader.data.get("iio_us-east-1_i-a2eb1cd9_NetworkIn.csv")

"""
Capire stationarity:
- Time plot -> capire se c'è trend/seasonality/necessità log transform
- ACF, PACF e Box-Ljung test -> da qui si vede se c'è autocorrelazione
- Test statistici per capire stazionarità: KPSS. Applicazione ripetuta dà
  numero differenziazioni che sono da applicare.

Differenziazione per stazionarizzare
- Dopo eventuale log, fare sesonal differencing se necessario e poi
  differencing normale

Capire valori p e q per AR e MA:
- Usare ACF-PACF (solo se dati vengono da ARIMA(p,d,0) o ARIMA(0,d,q))
- Usare AIC_c per scegliere modello migliore
- Controllare residuals (ACF, portmanteau)
- Calcolare forecast quando residuals sono soddisfacendti
"""

# df.plot()
# plt.show()
train_percentage = 1
train_len = int(1 * len(df))
data = df.value.values  # / df.value.values.max()
print(tsa.kpss(data, lags='auto'))
train, test = data[:train_len], data[train_len:]

auto_fit = False
if auto_fit:
    arima = auto_arima(train, stepwise=True, trace=1, seasonal=False)
    print(arima.summary())
else:
    arima = ARIMA(order=(4, 1, 4), seasonal_order=None)
    arima.fit(train)

fitted_values = arima.predict_in_sample()
plt.plot(df.index[:-1], fitted_values, color='C0')
plt.plot(df.index, data, color='C1')
plt.plot(df.index[:-1], arima.resid(), color='C2')
plt.gca().grid(which='both', axis='x', linestyle='--')
print("Err: {}".format((arima.resid() ** 2).sum()))

plt.show()
print("..Exiting")
sys.exit(1)
predictions = arima.predict(n_periods=test.shape[0])
# ax.plot(df.index[train_len:], predictions)
plt.figure()
plt.plot(df.index[train_len:], test, '--', color='C0', label="test set")
plt.plot(df.index[train_len:], predictions,
         '--', color='C1', label="predictions")
plt.plot(df.index[:train_len], train, color='C0', label="train set")
plt.plot(df.index[:train_len - 1], fitted_values,
         color='C1', label="fitted values")
plt.legend()
arima.plot_diagnostics()
plt.show()
