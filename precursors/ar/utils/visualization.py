from pandas.plotting import lag_plot, autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
import pandas as pd


def time_series_data_visualization(dataset, date_columns, target_value):
    plt.figure(figsize=(10, 5))
    plt.plot(pd.to_datetime(dataset[date_columns]), dataset[target_value], label=target_value)
    plt.legend()
    plt.show()
    plt.close()


def lag_plot_visualization(dataset, target_value):
    lag_plot(dataset[target_value])
    plt.show()
    plt.close()


def auto_correlation_plot_visualization(dataset, target_value, length):
    # pandas auto correlation graph
    autocorrelation_plot(dataset[target_value])

    for i in range(length):
        plt.axvline(x=365 * i, color='r', linestyle='--', linewidth=2)

    plt.show()
    plt.close()

    # statsmodels auto correlation graph
    plot_acf(dataset[target_value])
    plt.ylim([-0.5, 1.5])
    plt.show()
    plt.close()


def auto_regressive_plot_visualization(test, prediction):
    plt.figure(figsize=(10, 5))
    plt.plot(test)
    plt.plot(prediction)
    plt.show()
    plt.close()
