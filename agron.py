import decimal
from boto3 import resource
from dataclasses import dataclass
from decimal import Decimal
from dotenv import load_dotenv
from os import getenv
from s3_helper import CSVStream
from typing import Any
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dropout
from keras.preprocessing.sequence import TimeseriesGenerator

def response_fill(price: decimal.Decimal, volume: decimal.Decimal, error_code:str, error_msg: str, unfilled: dict[str, decimal.Decimal]):
    x = dict()
    x['price'] = price
    x['volume'] = volume
    x['error_code'] = error_code
    x['error_msg'] = error_msg
    x['unfilled'] = unfilled
    return x

if __name__ == '__main__':
    MMS = MinMaxScaler()
    data = pd.read_pickle('mainData.pkl')
    data = data[['price', 'time']]
    data['time'] = pd.to_numeric(data['time'])
    data = data[data.time > 15000000]
    data['time'] = pd.to_datetime(data['time'], unit='s')
    #remove data from pandas that has time 0
    data = data.drop_duplicates(subset=['time'])
    data['price'] = data['price'].astype(float)
    data = data.sort_values(by='time')
    data.to_csv('mainData.csv', index=False)
    #print(data.head())
    #print(data.dtypes)
    #data.plot(data['time'], data['price'])
    #data.plot(kind='scatter',x='time', y='price')
    #plt.xticks(rotation='vertical')
    #plt.show()
    preprocess_prices = data['price'].values
    preprocess_prices = preprocess_prices.reshape((-1,1))

    training_split = 0.80
    splitted = int(training_split*len(preprocess_prices))
    train_data = preprocess_prices[:splitted]
    test_data = preprocess_prices[splitted:]
    training_dates = data['time'][:splitted]
    test_dates = data['time'][splitted:]

    print(len(train_data))
    print(len(test_data))

    look_back = 50

    training_generator = TimeseriesGenerator(train_data, train_data, length=look_back, batch_size=20)
    testing_generator = TimeseriesGenerator(test_data, test_data, length=look_back, batch_size=1)
    
    from keras.models import Sequential
    from keras.layers import LSTM, Dense

    print(train_data)

    model = Sequential()
    model.add(LSTM(42, activation='relu', input_shape=(look_back, 1)))

    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    num_epochs = 50
    #model.fit(training_dates, train_data, epochs=num_epochs, verbose=1)
    model.fit_generator(training_generator, epochs=num_epochs, verbose=1)

    prediction = model.predict_generator(testing_generator)

    price_train = train_data.reshape((-1))
    price_test = test_data.reshape((-1))
    prediction = prediction.reshape((-1))

    