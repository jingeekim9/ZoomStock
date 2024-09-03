import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def compute_coefficients(symbols, prices):
    coefs = []
    for i1, s1, p1 in zip(range(len(symbols)), symbols, prices):
        for i2, s2, p2 in zip(range(len(symbols)), symbols, prices):
            if i1 < i2:
                coef = np.corrcoef(np.stack([p1, p2]))[0, 1]
                coefs.append((s1, s2, coef))
    return sorted(coefs, key=lambda x: -x[2])


def main():
    path = '../../data/acl18/raw'
    symbols, dfs = [], []
    for file in os.listdir(path):
        symbols.append(file.split('.')[0])
        dfs.append(pd.read_csv(os.path.join(path, file), index_col='Date'))
    max_len = max(df.shape[0] for df in dfs)
    symbols = [symbol for i, symbol in enumerate(symbols) if dfs[i].shape[0] == max_len]
    dfs = [df for i, df in enumerate(dfs) if dfs[i].shape[0] == max_len]
    prices = [df.loc['2015-10-01':'2015-12-31', 'Adj Close'] for df in dfs]
    price_dict = {s: p / p[0] for s, p in zip(symbols, prices)}

    coefs = compute_coefficients(symbols, prices)

    # symbols = ['BBL', 'PTR', 'BHP', 'PICO']
    # symbols = ['AMZN', 'GOOG', 'MCD', 'UNH']
    # symbols = ['SLB', 'TOT', 'BP', 'IEP']
    # symbols = ['CELG', 'AMGN', 'DIS', 'WFC']
    # symbols = ['GD', 'CMCSA', 'LMT', 'MMM']
    symbols = ['FB', 'GOOG', 'AAPL', 'AMZN']
    # targets = ['GD']
    values = []

    for s1, s2, coef in coefs:
        if s1 in symbols and s2 in symbols:
            values.append(coef)
            print(s1, s2, coef)

    print(np.mean(values))

    fig = plt.figure(figsize=(5.5, 3))
    ax = fig.add_subplot()
    ax.set_xticks([0, 22, 42, 63])
    ax.set_ylabel('Relative price (starting from 1)')

    plots = []
    for s in symbols:
        prices = price_dict[s].reset_index()
        prices.set_index('Date', inplace=True)
        prices /= prices.iloc[0]
        plots.append(ax.plot(prices)[0])
    ax.legend(plots, symbols)
    path_out = f'../../out-fig'
    file_out = 'motivation.png'
    os.makedirs(path_out, exist_ok=True)
    fig.savefig(f'{path_out}/{file_out}', bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    main()
