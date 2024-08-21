from AR.utils.visualization import lag_plot_visualization, auto_correlation_plot_visualization

import pandas as pd


def acf_and_plotting(dataset, target_value, length):
    values = pd.DataFrame(dataset[target_value].values)
    dataframe = pd.concat([dataset[target_value], values], axis=1)
    dataframe.columns = ['t-1', 't+1']
    result = dataframe.corr()

    print(result)

    lag_plot_visualization(dataset, target_value)
    auto_correlation_plot_visualization(dataset, target_value, length)
