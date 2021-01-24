# Filename: p1.py
# Author: Jacob Lu

# %%
import numpy as np
import pandas as pd 

# %%
def parseSeries(series_to_parse: pd.Series):
    """
    Input a pandas Series and return a list of 'Ideas'.
    """
    # Get an empty list of 'Ideas'.
    idea_list = list()

    # Record the previous quantity.
    prev_quantity = 0
    # Maintain a list of enter dates and a list of exit dates for further use.
    enter_dates = list()
    exit_dates = list()
    # Iterate over the given series and append to idea list.
    for d, quantity in series_to_parse.items():
        if quantity != 0 and prev_quantity == 0:
            enter_dates.append(d)
        elif quantity == 0 and prev_quantity != 0:
            exit_dates.append(d)
            idea_list += [series_to_parse[enter_dates[-1]:exit_dates[-1]]]
        prev_quantity = quantity

    # If a position does not exit at the end, force it to exit, and append it to idea list.
    if len(enter_dates) != len(exit_dates):
        idea_list.append(series_to_parse[enter_dates[-1]:])
    return idea_list

# %%
def parseDf(df_to_parse: pd.DataFrame):
    """
    Iterate over DataFrame column by column and append 'Ideas' to a return list.
    'Ideas' will be aggregated by stock names.
    """
    idea_list = list()
    for col in df_to_parse:
        idea_list = idea_list + [(col, parseSeries(df_to_parse[col]))]
    return idea_list

# %%
from pandas.testing import assert_series_equal
import pytest

# %%
def test_parseSeries1():
    series = pd.Series([0,1,2,3,0], index=pd.date_range("2020-12-14", "2020-12-18"))
    expected = [pd.Series([1,2,3,0], index=pd.date_range("2020-12-15", "2020-12-18"))]
    output = parseSeries(series)
    assert len(output) == len(expected)
    for i in range(len(output)):
        assert_series_equal(output[i], expected[i])

def test_parseSeries2():
    series = pd.Series([0,0,0,0,0], index=pd.date_range("2020-12-14", "2020-12-18"))
    expected = []
    output = parseSeries(series)
    assert len(output) == len(expected)

def test_parseSeries3():
    series = pd.Series([0,1,2,0,3,0], index=pd.date_range("2020-12-14", "2020-12-19"))
    expected = [pd.Series([1,2,0], index=pd.date_range("2020-12-15", "2020-12-17")),
                pd.Series([3,0], index=pd.date_range("2020-12-18", "2020-12-19"))]
    output = parseSeries(series)
    assert len(output) == len(expected)
    for i in range(len(output)):
        assert_series_equal(output[i], expected[i])

def test_parseSeries4():
    series = pd.Series([1,2,0,3,0], index=pd.date_range("2020-12-14", "2020-12-18"))
    expected = [pd.Series([1,2,0], index=pd.date_range("2020-12-14", "2020-12-16")),
                pd.Series([3,0], index=pd.date_range("2020-12-17", "2020-12-18"))]
    output = parseSeries(series)
    assert len(output) == len(expected)
    for i in range(len(output)):
        assert_series_equal(output[i], expected[i])

def test_parseSeries5():
    series = pd.Series([0,1,2,0,3], index=pd.date_range("2020-12-14", "2020-12-18"))
    expected = [pd.Series([1,2,0], index=pd.date_range("2020-12-15", "2020-12-17")),
                pd.Series([3], index=pd.date_range("2020-12-18", "2020-12-18"))]
    output = parseSeries(series)
    assert len(output) == len(expected)
    for i in range(len(output)):
        assert_series_equal(output[i], expected[i])

def test_parseDf():
    series1 = pd.Series([0,1,2,3,0], index=pd.date_range("2020-12-14", "2020-12-18"))
    series2 = pd.Series([0,1,0,2,0], index=pd.date_range("2020-12-14", "2020-12-18"))
    df = pd.DataFrame({"000001": series1, "000002": series2})
    output = parseDf(df)
    expected = [
        ("000001", [pd.Series([1, 2, 3, 0], index=pd.date_range("2020-12-15", "2020-12-18"), name="000001")]),
        ("000002", [pd.Series([1, 0], index=pd.date_range("2020-12-15", "2020-12-16"), name="000002"),
        pd.Series([2, 0], index=pd.date_range("2020-12-17", "2020-12-18"), name="000002")])
    ]
    assert len(output) == len(expected)
    for i in range(len(output)):
        assert len(output[i]) == 2
        assert output[i][0] == expected[i][0]
        for j in range(len(output[i][1])):
            assert_series_equal(output[i][1][j], expected[i][1][j])
