import pandas as pd
import numpy as np
from pandas import DataFrame


def log_column(data: DataFrame, column: str) -> DataFrame:
    """

    :param data: DataFrame with values to log
    :param column: Column to log
    :return: DataFrame with log column
    """
    data = np.log1p(data[column])
    return data


def drop_column(data: DataFrame, column: str, axis: int) -> DataFrame:
    """

    :param data: DataFrame with values to drop column
    :param column: Column to drop
    :param axis: Axis to drop. Column = -1 Row = 0
    :return: DataFrame without chosen column.
    """
    data = data.drop(column, axis=axis)
    return data
