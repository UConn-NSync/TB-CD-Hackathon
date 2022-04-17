import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# this is important to get the correct path

if __name__ == '__main__':
    data = pd.read_pickle('mainData.pkl')
    data['price'] = data['price'].astype(float)
    mainData = pd.DataFrame()
    mainData['BTC'] = data['price'].values
    print(mainData)

    x = data['time'].values
    y = data['price'].values
    offset = int(0.25 * len(data))

    x_train = x[:-offset]
    y_train = y[:-offset]
    x_test = x[-offset:]
    y_test = y[-offset:]


    from statsmodels.tsa.arima.model import ARIMA
    model = ARIMA(y_train, order=(5, 1, 0)).fit()
    forecast = model.forecast(steps=1000)

    print(y_test[-1])
    print(forecast[0])

    

    plt.plot(range(0, len(y_train)), y_train, 'r', label='Training data')
    plt.plot(range(len(y_train), len(y)), y_test, 'b', label='Test data')
    plt.plot(range(len(y), len(y) + len(forecast)), forecast, 'g', label='Predicted data')

    plt.show()


