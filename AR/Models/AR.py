from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error, mean_absolute_error
from ..utils.visualization import auto_regressive_plot_visualization


def auto_regressive(dataset, target_value, days, count, lag_value):
    values = dataset[target_value]
    train, test = values[:len(values) - (days * count)], values[len(values) - (days * count):]

    # lags = creating coefficient counts
    model = AutoReg(train, lags=lag_value)
    model_fit = model.fit()

    print(model_fit.summary())
    return model_fit, train, test


def predict_value(model, train, test):
    prediction = model.predict(start=len(train), end=len(train) + len(test)-1, dynamic=False)

    print('MSE : ', mean_squared_error(test, prediction))
    print('MAE : ', mean_absolute_error(test, prediction))

    auto_regressive_plot_visualization(test, prediction)