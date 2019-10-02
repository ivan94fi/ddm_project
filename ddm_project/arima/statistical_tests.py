"""
This module contains various statistical hypothesis tests.

Contains:
- Kwiatkowski–Phillips–Schmidt–Shin (KPSS) test and augmented Dickey–Fuller
  (ADF) test to test the stationarity of a time-series.
- Ljung–Box test to test the autocorrelation of a time series.
"""

from collections import namedtuple

import statsmodels.tsa.api as tsa

KPSSResults = namedtuple(
    "KPSSResults", ["kpss_stat", "p_value", "lags", "critical_values"])

# H0 HA lags nobs in comune

TestResults = namedtuple(
    "TestResults", ["test_name", "test_stat", "p_value", "crit_values"])

"""
test_stat: adf
"""


def kpss(x, **kwargs):
    """Compute KPSS test for stationarity."""
    res = tsa.kpss(x, nlags='auto', **kwargs)
    test_results = TestResults(
        test_name="KPSS", test_stat=res[0], p_value=res[1], crit_values=res[3])
    return test_results


def adf(x, **kwargs):
    """Compute ADF test for stationarity."""
    res = tsa.adfuller(x, **kwargs)
    test_results = TestResults(
        test_name="ADF", test_stat=res[0], p_value=res[1], crit_values=res[4])
    return test_results


def box_ljung(x, nobs, **kwargs):
    """Compute Box-Ljung Q statistic for autocorrelation."""
    res = tsa.q_stat(x, nobs, **kwargs)
    test_results = TestResults(test_name="Box-Ljung", test_stat=res[0],
                               p_value=res[1], crit_values=None)
    return test_results


if __name__ == '__main__':
    import numpy as np

    x = np.random.randn(100)
    # df = pd.read_csv("~/goog.csv", usecols=[1])
    # x = df.x

    kpss_results = kpss(x)
    adf_results = adf(x)
    box_ljung_results = box_ljung(x, len(x))

    # pprint(kpss_results)
    # pprint(adf_results)
    # pprint(box_ljung_results)
