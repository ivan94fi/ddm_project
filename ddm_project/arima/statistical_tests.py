"""
This module contains various statistical hypothesis tests.

Contains:
- Kwiatkowski–Phillips–Schmidt–Shin (KPSS) test and augmented Dickey–Fuller
  (ADF) test to test the stationarity of a time-series.
- Ljung–Box test to test the autocorrelation of a time series.
"""

from typing import NamedTuple

import statsmodels.tsa.api as tsa
from tabulate import tabulate


class TestResults(NamedTuple):
    """NamedTuple to hold tests results."""

    test_name: str
    test_stat: float
    p_value: float
    lags: int = None
    crit_values: dict = None

    def format(self):
        """Format results for printing."""
        tabular_data = [
            ["Test", self.test_name],
            ["Value of test-statistic", self.test_stat],
            ["p-value", self.p_value],
            ["Lags", self.lags],
            ["Critical values", self.crit_values],
        ]
        formatted_table = tabulate(tabular_data, colalign=("right",))
        table_width = formatted_table.index("\n")
        print("-" * table_width)
        title = "{} test results".format(self.test_name)
        padding = (table_width - len(title)) // 2
        print(" " * padding + title)
        print(formatted_table)


def kpss(x, **kwargs):
    """Compute KPSS test for stationarity."""
    res = tsa.kpss(x, nlags="auto", **kwargs)
    test_results = TestResults(
        test_name="KPSS",
        test_stat=res[0],
        p_value=res[1],
        lags=res[2],
        crit_values=res[3]
    )
    return test_results


def adf(x, **kwargs):
    """Compute ADF test for stationarity."""
    res = tsa.adfuller(x, **kwargs)
    test_results = TestResults(
        test_name="ADF",
        test_stat=res[0],
        p_value=res[1],
        lags=res[2],
        crit_values=res[4]
    )
    return test_results


def box_ljung(x, **kwargs):
    """Compute Box-Ljung Q statistic for autocorrelation."""
    res = tsa.acf(x, qstat=True, fft=True, ** kwargs)
    test_results = TestResults(
        test_name="Box-Ljung",
        test_stat=res[1],
        p_value=res[2]
    )
    return test_results


if __name__ == '__main__':
    import numpy as np
    np.random.seed(42)
    x = np.random.randn(100)

    # import pandas as pd
    # df = pd.read_csv("~/goog.csv", usecols=[1])
    # x = df.x[:200]

    kpss_results = kpss(x)
    adf_results = adf(x, maxlag=40)
    box_ljung_results = box_ljung(x, nlags=20)

    # pprint(kpss_results)
    # pprint(adf_results)
    # pprint(box_ljung_results)

    # kpss_results.format()
    # adf_results.format()
    # box_ljung_results.format()

    b = box_ljung_results
    b.format()
    # print("")
    #
    # for i in range(20):
    #     print("{: 12.5f} {}".format(b.test_stat[i], b.p_value[i]))
