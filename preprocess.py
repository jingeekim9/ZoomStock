"""
ZoomStock (BigData 2024)

Authors:
    - JinGee Kim (jingeekim9@snu.ac.kr)
    - Yong-chan Park (wjdakf3948@snu.ac.kr)
    - Jaemin Hong (jmhong0120@snu.ac.kr)
    - U Kang (ukang@snu.ac.kr)

Affiliation:
    - Data Mining Lab., Seoul National University

File: models.py
     - Generate new datasets.

Version: 1.0.0
"""

import pandas as pd
import datetime
import pymysql
import os
import numpy as np


def save_raw_stocks(path, con, date_from, date_to):
    """
    Loads stock prices from the DeepTrade database and saves them as a CSV file.

    :param path: Path to store the loaded prices.
    :param con: Connector to the database.
    :param date_from: The start date for loading data.
    :param date_to: The end date for loading data.
    :return: None.
    """
    # Query to find the latest date in the stocks_info_us table.
    sql = 'select max(date) from stocks_info_us'
    date = pd.read_sql_query(sql, con).iloc[0, 0]
    
    # Query to fetch symbols for NDX-listed stocks as of the latest date.
    sql = 'select symbol from stocks_info_us where ndx = 1 and date = %s'
    symbols = pd.read_sql_query(sql, con, params=[date])['symbol'].tolist()
    
    # Query to fetch stock prices for the selected symbols within the date range.
    sql = 'select distinct * from stocks_us where symbol in %s'
    df = pd.read_sql_query(sql, con, params=[symbols])
    df = df[(df['date'] >= date_from) & (df['date'] <= date_to)]
    
    # Save the result as a CSV file.
    df.to_csv(path, index=False)


def save_jpn_stocks(path, con, date_from, date_to):
    """
    Loads Japan stock prices and saves them to a CSV file.
    
    :param path: Path to store the loaded prices.
    :param con: Connector to the database.
    :param date_from: Start date for loading data.
    :param date_to: End date for loading data.
    :return: None.
    """
    print(path)
    sql = 'select max(date) from stocks_info_jpn'
    date = pd.read_sql_query(sql, con).iloc[0, 0]
    sql = 'select symbol from stocks_info_jpn where date = %s'
    symbols = pd.read_sql_query(sql, con, params=[date])['symbol'].tolist()
    sql = 'select distinct * from stocks_jpn where symbol in %s'
    df = pd.read_sql_query(sql, con, params=[symbols])
    df = df[(df['date'] >= date_from) & (df['date'] <= date_to)]
    df.to_csv(path, index=False)


# Similar functions for different regions, e.g., `save_gb_stocks`, `save_chn_stocks`,
# `save_crypto_stocks`, `save_kospi_stocks`, and `save_kosdaq_stocks`.
# These functions all load stock prices for a specific market or region and save them as a CSV file.


def save_symbols(df, path):
    """
    Saves a list of symbols that have sufficient data points to a CSV file.

    :param df: DataFrame representing a dataset.
    :param path: Path to store the symbols.
    :return: None.
    """
    valid_symbols = []
    for symbol, df_ in df.groupby('symbol'):
        # Check if the symbol has more than or equal to 502 data points.
        n_points = df[df['symbol'] == symbol].shape[0]
        if n_points >= 502:
            valid_symbols.append(symbol)
    df_symbols = pd.DataFrame(valid_symbols, columns=['symbol'])
    df_symbols.to_csv(path, index=False)


def make_features(df):
    """
    Generates feature vectors from raw stock prices.

    :param df: DataFrame containing raw stock prices.
    :return: Numpy array of generated features.
    """
    def to_label(x):
        # Convert the price change into a label.
        if x >= 0.55:
            return 1
        elif x < -0.5:
            return -1
        else:
            return 0

    adj_closes = []
    for i in range(30):
        adj_closes.append(df['Adj Close'].shift(periods=i).values)
    adj_closes = np.stack(adj_closes, axis=0)
    df_shifted = df.shift()

    # Calculate various features such as opening and closing price ratios.
    c_open = df['Open'] / df['Close'] - 1
    c_high = df['High'] / df['Close'] - 1
    c_low = df['Low'] / df['Close'] - 1
    n_close = df['Close'] / df_shifted['Close'] - 1
    n_adj_close = df['Adj Close'] / df_shifted['Adj Close'] - 1
    day_5 = adj_closes[:5].mean(axis=0) / df['Adj Close'] - 1
    day_10 = adj_closes[:10].mean(axis=0) / df['Adj Close'] - 1
    day_15 = adj_closes[:15].mean(axis=0) / df['Adj Close'] - 1
    day_20 = adj_closes[:20].mean(axis=0) / df['Adj Close'] - 1
    day_25 = adj_closes[:25].mean(axis=0) / df['Adj Close'] - 1
    day_30 = adj_closes[:30].mean(axis=0) / df['Adj Close'] - 1
    label = (df['Adj Close'] / df_shifted['Adj Close'] - 1) * 100
    label = label.apply(to_label)
    adj_close = df['Adj Close']

    values1 = [c_open, c_high, c_low, n_close, n_adj_close, day_5, day_10, day_15, day_20, day_25, day_30]
    values1 = np.stack(values1, axis=1) * 100
    values2 = np.stack([label, adj_close], axis=1)
    features = np.concatenate([values1, values2], axis=1).astype(np.float32)
    features[:29] = -123321
    return features


def generate_new_dataset(name):
    """
    Generates a new dataset and saves stock price and symbol data.

    :param name: Name of the dataset to generate.
    :return: None.
    """
    con = pymysql.connect(host='klimt1.snu.ac.kr',
                          port=13306,
                          user='deeptrade',
                          password='123456qwer!',
                          db='deeptrade')

    path = f'../data/{name}'
    path_stocks = f'{path}/raw/stocks.csv'
    os.makedirs(os.path.dirname(path_stocks), exist_ok=True)
    if not os.path.exists(path_stocks):
        # Select appropriate function and date range based on the dataset name.
        if name == 'ndx-short':
            date_from = datetime.date(2017, 1, 1)
            date_to = datetime.date(2019, 1, 1)
            save_raw_stocks(path_stocks, con, date_from, date_to)
        # Other datasets with specified date ranges
        # ...

    df = pd.read_csv(path_stocks)
    df['Open'] = df['opening_price']
    df['High'] = df['highest_price']
    df['Low'] = df['lowest_price']
    df['Close'] = df['closing_price']
    df['Adj Close'] = df['closing_price'] / df['adjusting_factor']
    df.drop(columns=['opening_price', 'highest_price', 'lowest_price', 'closing_price', 'trading_volume', 'adjusting_factor'], inplace=True)

    # Save trading dates for symbol '000070'.
    df[df['symbol'] == '000070']['date'].to_csv(f'{path}/trading_dates.csv', header=False, index=False)

    # Save valid symbols to CSV.
    path_symbols = f'{path}/raw/symbols.csv'
    if not os.path.exists(path_symbols):
        save_symbols(df, path_symbols)
    symbols = set(pd.read_csv(path_symbols)['symbol'].tolist())

    max_len = df.groupby('symbol').size().max()

    os.makedirs(f'{path}/ourpped', exist_ok=True)
    for symbol, df_ in df.groupby('symbol'):
        if symbol in symbols:
            df_ = df_.sort_values('date')
            x = make_features(df_)
            padding = np.full((max_len - x.shape[0], x.shape[1]), -123321.000000)
            x = np.concatenate([padding, x])
            path_features = f'{path}/ourpped/{symbol}.csv'
            np.savetxt(path_features, x, delimiter=',', fmt='%.6f')


def preprocess_index(index_name):
    """
    Loads and saves trading dates and feature vectors for an index.

    :param index_name: Name of the index to generate.
    :return: None.
    """
    df = pd.read_csv('../data/index/{}/raw.csv'.format(index_name))
    path = '../data/index/{}/trading_dates.csv'.format(index_name)
    df['Date'].to_csv(path, index=False, header=False)
    df = df.set_index('Date')
    x = make_features(df)
    path = '../data/index/{}/ourpped.csv'.format(index_name)
    np.savetxt(path, x, delimiter=',', fmt='%.6f')


def preprocess_index_for_dataset(target, index_name='snp500'):
    """
    Preprocesses and saves index data for each dataset.

    :param target: Dataset name to save the data in.
    :param index_name: Index name to generate data for.
    :return: None.
    """
    df_index = pd.read_csv('../data/index/{}/raw.csv'.format(index_name))
    df_target = pd.read_csv('../data/{}/raw/stocks.csv'.format(target))
    df_index = df_index.set_index('Date')
    x = make_features(df_index)
    path = '../data/{}/ourpped/{}.csv'.format(target, index_name)
    np.savetxt(path, x, delimiter=',', fmt='%.6f')



def main():
    """
    Main function that generates datasets.

    :return: None.
    """
    # preprocess_index('nasdaq')
    # preprocess_index('snp500')
    # preprocess_index('nikkei')
    # preprocess_index('ftse')
    preprocess_index('kospi200')
    # preprocess_index('csi300')
    # preprocess_index('ubmi')

    # generate_new_dataset(name='jpn20')
    # generate_new_dataset(name='gb20')
    generate_new_dataset(name='kospi')
    # generate_new_dataset(name='chn20')
    # generate_new_dataset(name='crypto')
    #generate_new_dataset(name='ndx-short')
    #generate_new_dataset(name='ndx-long')

    # preprocess_index_for_dataset(target='jpn20', index_name='nikkei')
    # preprocess_index_for_dataset(target='chn20', index_name='csi300')
    # preprocess_index_for_dataset(target='crypto', index_name='ubmi')
    preprocess_index_for_dataset(target='kospi', index_name='kospi200')
    # preprocess_index_for_dataset(target='gb20', index_name='ftse')
    # preprocess_index_for_dataset(target='ndx-short')
    # preprocess_index_for_dataset(target='ndx-long')


if __name__ == '__main__':
    main()
