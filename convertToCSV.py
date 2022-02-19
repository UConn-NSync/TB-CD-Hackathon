import pandas as pd

if __name__ == '__main__':
    print("reading data!")
    data = pd.read_csv("~/Downloads//xbt.usd.2018", header=None, names=['exchange', 'price', 'volume', 'time'])
    print("done reading")
    data = data[data['exchange'] == 'bfnx-xbt-usd']
    print(data)
    print(len(data))
    data = data[['price', 'time']]
    data['time'] = pd.to_numeric(data['time'])
    data['price'] = data['price'].astype(float)
    data = data[data.time > 15000000]
    data = data.sort_values(by='time')
    data = data.drop_duplicates(subset=['time'])
    print(data)
    data.to_csv('testing_data_bfnx.csv', index=False)