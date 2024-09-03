import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def compute_coefficients(target, symbols, all_symbols, prices):
    target_idx = all_symbols.index(target)
    target_prices = prices[target_idx]
    coefs = []
    for s in symbols:
        p = prices[all_symbols.index(s)]
        coef = np.corrcoef(np.stack([target_prices, p]))[0, 1]
        coefs.append(coef)
    return coefs


def main():
    path = '../../data/acl18/raw'
    symbols, dfs = [], []
    for file in os.listdir(path):
        symbols.append(file.split('.')[0])
        dfs.append(pd.read_csv(os.path.join(path, file), index_col='Date'))
    max_len = max(df.shape[0] for df in dfs)
    all_symbols = [symbol for i, symbol in enumerate(symbols) if dfs[i].shape[0] == max_len]
    dfs = [df for i, df in enumerate(dfs) if dfs[i].shape[0] == max_len]

    target = 'FB'
    symbols = ['AMZN', 'AAPL', 'GOOG', 'MSFT']
    markers = ['o', 'v', '^', 's']
    coef_list = []
    for year in range(2013, 2018):
        prices = [df.loc[f'{year}-01-01':f'{year}-12-31', 'Adj Close'] for df in dfs]
        coefs = compute_coefficients(target, symbols, all_symbols, prices)
        coef_list.append(coefs)
    data = np.array(coef_list).transpose()

    fig = plt.figure(figsize=(4.5, 2.5))
    ax = fig.add_subplot()
    plots = []
    for i in range(len(symbols)):
        plots.append(ax.plot(data[i], 'o-', marker=markers[i])[0])
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels([2013, 2014, 2015, 2016, 2017])
    ax.set_ylabel('Pearson correlation coefficient')

    # plots = []
    # for s in symbols:
    #     prices = price_dict[s].reset_index()
    #     prices.set_index('Date', inplace=True)
    #     prices /= prices.iloc[0]
    #     plots.append(ax.plot(prices)[0])
    ax.legend(plots, symbols)
    path_out = f'../../out-fig'
    file_out = 'motivation.png'
    os.makedirs(path_out, exist_ok=True)
    fig.savefig(f'{path_out}/{file_out}', bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    main()
