from pandas import DataFrame
from sklearn import metrics
import numpy as np
import pandas as pd


def _prepare_score(y_test: DataFrame, y_pred: DataFrame) -> DataFrame:
    """

    :rtype: object
    """
    data = [metrics.mean_absolute_error(y_test, y_pred),
            metrics.mean_squared_error(y_test, y_pred),
            np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
            metrics.r2_score(y_test, y_pred)]

    data = pd.DataFrame([data],
                        columns=['mae', 'mse', 'rmse', 'r2'])
    return data


def print_score(y_test: DataFrame, y_pred: DataFrame):
    """

    :param y_test: DataFrame with target values
    :param y_pred: DataFrame with predicted values
    """
    scores = _prepare_score(y_test, y_pred)
    print('R2: ', scores.r2.values)
    print('Mean Absolute Error:', scores.mae.values)
    print('Mean Squared Error:', scores.mse.values)
    print('Root Mean Squared Error:', scores.rmse.values)


def create_result_dataframe(y_test: DataFrame, y_pred: DataFrame, model_name: str) -> DataFrame:
    """

    :param y_test: DataFrame with target values
    :param y_pred: DataFrame with predicted values
    :param model_name: Actual model name
    :return: DataFrame with models scores
    """
    scores = _prepare_score(y_test, y_pred)
    data = pd.DataFrame(columns=['model'])
    data = data.append(pd.DataFrame(
        {
            'model': model_name,
            'R2': scores.r2.values,
            'Mean Absolute Error': scores.mae.values,
            'Mean Squared Error': scores.mse.values,
            'Root Mean Squared Error': scores.rmse.values
        }, index=[0]
    ))
    return data
