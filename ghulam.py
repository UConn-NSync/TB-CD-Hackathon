from time import time_ns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
import datetime as dt 
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM

 
r = pd.read_pickle("mainData.pkl")

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(r["time"].values.reshape(-1,1))

# how far to look back to predict the next days data 
prediction_day = 9097

# arrays to hold informartion of the data
price_arr = []
time_arr = []

def prepare_data(r):
  
    x = pd.Series(r["time"])
    time_arr = x.to_numpy(dtype='int64')

    y = pd.Series(r["price"])
    price_arr = y.to_numpy(dtype='float64')

    return time_arr, price_arr


def reshape_data(r):
    k, l = prepare_data(r)
    p, h = k.shape, l.shape

    return p, h