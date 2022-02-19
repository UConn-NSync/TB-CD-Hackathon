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
from ghulam import prepare_data, reshape_data
from agron import * 

load_dotenv()

BUY = "buy"
SELL = "sell"

BUCKET = getenv("BUCKET_NAME")

XBT_2018_KEY = "xbt.usd.2018"
XBT_2020_KEY = "xbt.usd.2020"

ETH_2018_KEY = "eth.usd.2018"
ETH_2020_KEY = "eth.usd.2020"

S3 = resource("s3")
SELECT_ALL_QUERY = 'SELECT * FROM S3Object'
#Jan 1 2018 to Jan 7 2019
SELECT_Jan_Jan = 'SELECT * FROM S3Object s WHERE s._1 = \'okf0-xbt-usd\' AND CAST(s._4 AS DECIMAL) >= 1514836800 AND CAST(s._4 AS DECIMAL) < 1514844000'

# Example s3 SELECT Query to Filter Data Stream
#
# The where clause fields refer to the timestamp column in the csv row.
# To filter the month of February, for example, (start: 1517443200, end: 1519862400) 2018
#                                               (Feb-01 00:00:00  , Mar-01 00:00:00) 2018
#
# QUERY = '''\
#     SELECT *
#     FROM S3Object s
#     WHERE CAST(s._4 AS DECIMAL) >= 1514764800
#       AND CAST(s._4 AS DECIMAL) < 1514764802
# '''

STREAM = CSVStream(
    'select',
    S3.meta.client,
    key=XBT_2018_KEY,
    bucket=BUCKET,
    expression=SELECT_Jan_Jan,
)

@dataclass
class Trade:
    trade_type: str # BUY | SELL
    base: str
    volume: Decimal

def response_fill(price: decimal.Decimal, volume: decimal.Decimal, unfilled: dict[str, decimal.Decimal], error_code:str=None, error_msg: str=None):
    x = dict()
    x['price'] = price
    x['volume'] = volume
    if(error_code):
        x['error_code'] = error_code
    if(error_msg):
        x['error_msg'] = error_msg
    x['unfilled'] = unfilled
    return x

def fill():


def algorithm(csv_row: str, context: dict[str, Any]):
    """ Trading Algorithm

    Add your logic to this function. This function will simulate a streaming
    interface with exchange trade data. This function will be called for each
    data row received from the stream.

    The context object will persist between iterations of your algorithm.

    Args:
        csv_row (str): one exchange trade (format: "exchange pair", "price", "volume", "timestamp")
        context (dict[str, Any]): a context that will survive each iteration of the algorithm

    Generator:
        response (dict): "Fill"-type object with information for the current and unfilled trades
    
    Yield (None | Trade | [Trade]): a trade order/s; None indicates no trade action
    """

    # algorithm logic...


    #loads the data from the csv row
    csv_data = csv_row
    csv_data = csv_data.split(',')
    base = csv_data[0].split('-')[1]
    price = Decimal(csv_data[1])
    volume = Decimal(csv_data[2])
    timestamp = Decimal(csv_data[3])
    print(csv_data)

    future_price = calculate_future_price(timestamp, price, base)


    #if the price is lower than the future price, then attempt to buy
    if price+0.5 < future_price:
        #check if there is a previous trade
        if context['previous_trade'] != BUY: # there is no previous trade, so attempt to make buy signal
            print("ATTEMPT BUY ORDER")
            context['previous_trade'] = BUY
            
            #attempt to make a buy order
            
            t = Trade(BUY, 'xbt', 3) #only buy in chunks of 2-3btc

    #if the price is higher than the future price, then attempt to sell
    elif price-0.5 > future_price:
        #if there is no "Buy" trade before, then cannot sell.
        if context['previous_buys'] != BUY:
            #first check if we have btc or eth in our wallet


    
        

    # margin between the price and the future price, hodl

    response = yield t # example: Trade(BUY, 'xbt', Decimal(1))

    # algorithm clean-up/error handling...

    #t is the response data object with 

if __name__ == '__main__':
    context = {'currentMoney': Decimal(1000000), 'owned_xbt': Decimal(0), 'owned_eth': Decimal(0), 'previous_trade': None}


    pass

# Example Interaction
#
# Given the following incoming trades, each line represents one csv row:
#   (1) okfq-xbt-usd,14682.26,2,1514765115
#   (2) okf1-xbt-usd,13793.65,2,1514765115
#   (3) stmp-xbt-usd,13789.01,0.00152381,1514765115
#
# When you receive trade 1 through to your algorithm, if you decide to make
# a BUY trade for 3 xbt, the order will start to fill in the following steps
#   [1] 1 unit xbt from trade 1 (%50 available volume from the trade data)
#   [2] 1 unit xbt from trade 2
#   [3] receiving trade 3, you decide to put in another BUY trade:
#       i. Trade will be rejected, because we have not finished filling your 
#          previous trade
#       ii. The fill object will contain additional fields with error data
#           a. "error_code", which will be "rejected"; and
#           b. "error_msg", description why the trade was rejected.
#
# Responses during these iterations:
#   [1] success resulting in:
#       {
#           "price": 14682.26,
#           "volume": 1,
#           "unfilled": {"xbt": 2, "eth": 0 }
#       }
#   [2]
#       {
#           "price": 13793.65,
#           "volume": 1,
#           "unfilled": {"xbt": 1, "eth": 0 }
#       }
#   [3]
#       {
#           "price": 13789.01,
#           "volume": 0.000761905,
#           "error_code": "rejected",
#           "error_msg": "filling trade in progress",
#           "unfilled": {"xbt": 0.999238095, "eth": 0 }
#       }
#
# In step 3, the new trade order that you submitted is rejected; however,
# we will continue to fill that order that was already in progress, so
# the price and volume are CONFIRMED in that payload.


def calculate_future_price(timestamp: str, price: decimal.Decimal, base:str):
    """
    Calculates the future price of a given timestamp.
    """
    if 'xbt' in base:
        model.load("./models/xbt_model")
    elif 'eth' in base:
        model.load("./models/eth_model")
    else:
        print("Invalid base")
        return

    #convert timestamp to datetime
    timestamp = datetime.datetime.fromtimestamp(timestamp)

    tarry = np.array()
    #get the future price
    future_price = model.predict(tarray)

    