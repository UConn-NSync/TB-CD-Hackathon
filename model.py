import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import datetime
from datetime import datetime as dt
import pickle as pk
import tensorflow as tf
import tensorflow.keras.models as keras_models

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

data = pd.read_csv('/home/ubuntu/TB-CD-Hackathon/xbt_2018-2020_train.csv')

scaler = MinMaxScaler(feature_range=(0, 1))

#Train and save model
def train_model(df_col_name, pred_hours, save_path):

    scaled_data = scaler.fit_transform(data[df_col_name].values.reshape(-1,1))
    print(scaled_data)
    print(scaled_data.shape)
    x_train = []
    y_train = []

    for x in range(pred_hours, len(scaled_data)):
        x_train.append(scaled_data[x-pred_hours:x, 0])
        y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error', run_eagerly=True)
    model.fit(x_train, y_train, epochs=25, batch_size=20)
    model.save(save_path)
    print(f'Model saved to {save_path}!')


def predict_values(df_col_name, pred_hours, model_path, train_csv, test_csv):
    #scaler = MinMaxScaler(feature_range=(0, 1))
    #data = pd.read_csv(train_csv)
    #data1 = data
    #scaled_data = scaler.fit_transform(data1[df_col_name].values.reshape(-1,1))
    obj = scaler.fit(data[df_col_name].values.reshape(-1,1))
    model = keras_models.load_model(model_path)
    test_data = pd.read_csv(test_csv)
    actual_prices = data[df_col_name].values
    total_dataset = pd.concat((data[df_col_name], test_data[df_col_name]), axis=0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - pred_hours:].values
    model_inputs = model_inputs.reshape(-1,1)
    model_inputs = scaler.transform(model_inputs)

    x_test = []

    for x in range(pred_hours, len(model_inputs)):
        x_test.append(model_inputs[x-pred_hours:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    return predictions, actual_prices

def plot_preds(currency, dates, predictions, actual_prices):
    plt.plot(actual_prices, color="blue", label=f"Actual {currency} Price")
    plt.plot(predictions, color="orange", label=f"Predicted {currency} Price")
    plt.gca().invert_xaxis()
    plt.title(f"{currency} Price Over Time")
    plt.xlabel("Time")
    plt.ylabel(f"{currency} Price")
    plt.legend()
    plt.show()