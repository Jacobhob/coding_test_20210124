# Filename: p2.py
# Author: Jacob Lu

# %%
from math import floor
import numpy as np
import pandas as pd 
from scipy import stats

# %%
def correct(original: pd.Series, window=20, threshold_abs=2.5, fill='average'):
    """
    Find outliers in a series of number and return the corrected series and a list of outliers.
    Use a rolling frame to calculate the z-score of a data point. If z-score >= 2.5 or <= -2.5, mark an outlier.
    Dafault filling method is to take the average of adjacent data points.

    Arguments:
    window -- time frame for z-score, the data point will be in the middle of the window.
    threshold_abs -- a predetermined absolute threshold. Must be positive.
    fill -- Filling method. Default to be 'average'.
    """
    # Create an empty list for outliers.
    outliers = list()

    # Find if any outliers among the first [window/2] numbers. This is because the target number will be in the middle of the given window. Leaving the first [window/2] numbers unchecked.
    if original[0:floor(window/2)].std() != 0:
        if np.any(np.abs(stats.zscore(original[0:floor(window/2)])) >= threshold_abs):
            outliers += list(i for i in original[0:floor(window/2)][np.abs(stats.zscore(original[0:floor(window/2)])) >= threshold_abs].dropna().index.values)

    # Check z-score with a rolling window. Append to outliers list if one is found.
    zscore = ((original - original.rolling(window=window, min_periods=window, center=True).mean())/
        original.rolling(window=window, min_periods=window, center=True).std()).dropna()
    if np.any(np.abs(zscore) >= threshold_abs):
        outliers += list(i for i in zscore[np.abs(zscore) >= threshold_abs].dropna().index.values)

    # Same. Find if any outliers among the last [window/2] numbers.
    if original[-floor(window/2):].std() != 0:
        if np.any(np.abs(stats.zscore(original[-floor(window/2):])) >= threshold_abs):
            outliers += list(i for i in original[-floor(window/2):][np.abs(stats.zscore(original[-floor(window/2):])) >= threshold_abs].dropna().index.values)

    # Correct the given series. Fill the outliers with the average of adjacent data points.
    corrected = original.copy(deep=True)
    if fill == 'average':
        for d in outliers:
            pos = corrected.index.get_loc(d)
            corrected[d] = (corrected.iloc[pos-1] + corrected.iloc[pos+1])/2

    return corrected, outliers

# %%
from pandas.testing import assert_series_equal
import pytest

# %%
def test_correct1():
    series = pd.Series(range(366), index=pd.date_range("2020-01-01", "2020-12-31"))
    expected = [pd.Series(range(366), index=pd.date_range("2020-01-01", "2020-12-31")), []]
    output = correct(series)
    assert len(output) == len(expected)
    assert_series_equal(output[0], expected[0])
    assert output[1] == expected[1]

def test_correct2():
    series = pd.Series([0]*366, index=pd.date_range("2020-01-01", "2020-12-31"))
    expected = [pd.Series([0]*366, index=pd.date_range("2020-01-01", "2020-12-31")), []]
    output = correct(series)
    assert len(output) == len(expected)
    assert_series_equal(output[0], expected[0])
    assert output[1] == expected[1]

def test_correct3():
    series = pd.Series([1]*100+[10]+[1]*265, index=pd.date_range("2020-01-01", "2020-12-31"))
    expected = [pd.Series([1]*366, index=pd.date_range("2020-01-01", "2020-12-31")), [series.index[100]]]
    output = correct(series)
    assert len(output) == len(expected)
    assert_series_equal(output[0], expected[0])
    assert output[1] == expected[1]

def test_correct4():
    series = pd.Series([1]*5+[10]+[1]*354+[-10]+[1]*5, index=pd.date_range("2020-01-01", "2020-12-31"))
    expected = [pd.Series([1]*366, index=pd.date_range("2020-01-01", "2020-12-31")), [series.index[5], series.index[360]]]
    output = correct(series)
    assert len(output) == len(expected)
    assert_series_equal(output[0], expected[0])
    assert output[1] == expected[1]

def test_correct5():
    series = pd.Series([x**1.5 for x in range(50)], index=range(50))
    expected = [pd.Series([x**1.5 for x in range(50)], index=range(50)), []]
    output = correct(series)
    assert len(output) == len(expected)
    assert_series_equal(output[0], expected[0])
    assert output[1] == expected[1]
