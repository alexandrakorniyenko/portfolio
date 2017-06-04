import pandas as pd
import pandas_datareader.data as web
#import pandas.io.data as web  # Package and modules for importing data; this code may change depending on pandas version
import datetime
import pylab as py
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
import numpy as np
from portfolioopt import *
from dateutil.relativedelta import relativedelta
from stocks import *
from cvxopt import *
# start = datetime.datetime(2017, 5, 1)
# end = datetime.date.today()

# print(apple)
# py.rcParams['figure.figsize'] = (15, 9)  # Change the size of plots
# apple["Close"].plot(grid=True)
# plt.show()

# microsoft = web.DataReader("MSFT", "google", start, end)
# google = web.DataReader("GOOG", "google", start, end)
# apple = web.DataReader("AAPL", "google", start, end)
# intel = web.DataReader("INTL", "google", start, end)
#
# stocks = pd.DataFrame({"AAPL": apple["Close"],
#                        "MSFT": microsoft["Close"],
#                        "GOOG": google["Close"],
#                        'INTL': intel['Close']})
# print(stocks)
# print(stocks.cov())
# tickers = ['AAPl', 'IBM', 'MSFT', 'GOOG']
# df_list = []
# for ticker in tickers:
#     prices = web.DataReader(ticker, 'google', start, end)['Close']
#
#     df = pd.DataFrame({ticker: prices})
#     df_list.append(df)
#
#
# stocks = pd.concat(df_list, axis=1)
# stock_change = stocks.apply(lambda x: np.log(x) - np.log(x.shift(1)))
# returns = stocks.pct_change()
# avr_returns = stock_change.mean()
# cov = returns.cov()
# print(returns, avr_returns, cov)
# print(markowitz_portfolio(cov,avr_returns,0.0025))
def create_series(ticker):
    time_ago = datetime.datetime.today().date() - relativedelta(months = 12)
    ticker_data = web.get_data_google(ticker, time_ago)['Close'].pct_change().dropna()
    ticker_data_len = len(ticker_data)
    x = [0]*17
    y = [0]*17
    x_test = [0] * 17
    y_test = [0] * 17
    a = 0
    b = 6
    for i in range(0,17):
        x[i] = ticker_data[a:b]
        y[i] = ticker_data[b]
        a = b
        b = a + 6
    for i in range(0, 17):
        x_test[i] = ticker_data[a:b]
        y_test[i] = ticker_data[b]
        a = b
        b = a + 6
    return y,x,y_test,x_test

#print(create_series('AAPL'))
#ticker_data_acf = [ticker_data.autocorr(i) for i in range(1,32)]
# print(ticker_data_acf)
# print(np.argmin(np.absolute(ticker_data_acf)))
#
# test_df = pd.DataFrame(ticker_data_acf)
# test_df.columns = ['Autocorr']
# test_df.index += 1
# test_df.plot(kind='bar')
#plt.show()
