import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pandas import DataFrame
from scipy import stats


def visualize_score(data, viz_size: int, name: str):
    """

    :param name: Name of model. Needed for title.
    :param data: Data to visualize.
    :param viz_size: Number of data point to visualize.
    """
    df1 = data.head(viz_size)
    df1.plot(kind='bar', figsize=(16, 10))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.title(name)
    plt.show()


def visualize_distribution(data, column: str):
    """

    :param data: Data to visualize
    :param column: Column to plot distribution plot
    """
    sns.distplot(data[column])
    plt.title(column)
    plt.show()


def vizualize_correlation_matrix(data: DataFrame, x_size: int = 15, y_size: int = 15):
    """

    :param data: Data to visualize
    :param x_size: X size of visualization
    :param y_size: Y size of visualization
    """
    plt.figure(figsize=(x_size, y_size))
    sns.heatmap(
        data.corr(),
        vmax=1,
        vmin=-1,
        annot=True,
        square=True,
        fmt='.2f'
    )


def vizualize_prob_plot(data: DataFrame, title: str):
    """

    :param data: Data to visualize
    :param title: Title of visualization
    """
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
    stats.probplot(data, plot=ax[0])
    ax[0].set_title(title)
    stats.probplot(np.log1p(data), plot=ax[1])
    ax[1].set_title('log1p: ' + title)


def vizualize_catoregical_plot(data: DataFrame, measure: str):
    """

    :param data: Data to visualize
    :param measure: Name of measure to aggregate by catoregical columns
    """
    for feature in data.select_dtypes('object').columns:
        plt.figure(figsize=(20, 5))
        sns.barplot(data=data, x=feature, y=measure)
        plt.title(measure)
        plt.show()


def plot_scatter_score(y_test: DataFrame, y_pred: DataFrame, model_name: str, x_size: int = 10, y_size: int = 10):
    """

    :param y_test:
    :param y_pred:
    :param x_size:
    :param y_size:
    :param model_name:
    """
    plt.figure(figsize=(x_size, y_size))
    plt.scatter(y_pred, y_test, alpha=0.5, s=10)
    plt.plot((2, 20), (2, 20))
    plt.title('Evaluated predictions: ' + model_name, fontsize=15)
    plt.xlabel('Predictions')
    plt.ylabel('Test')
