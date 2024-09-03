import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def read_price_movements():
    path = '../data/acl18/raw'
    dfs = []
    for file in os.listdir(path):
        df = pd.read_csv(os.path.join(path, file), index_col='Date')
        df['tick'] = file.split('.')[0]
        df.reset_index(inplace=True)

        prices = df['Adj Close']
        returns = [0]
        for i in range(1, df.shape[0]):
            returns.append((prices[i] - prices[i - 1]) / prices[i - 1])
        df['pred'] = returns
        df['pred'] *= 100
        df.rename(columns=dict(Date='date'), inplace=True)
        dfs.append(df[['date', 'tick', 'pred']])
    return pd.concat(dfs)


def run_case_study(data, filename, ylim, ylabel):
    symbols = ['BAC', 'WFC', 'JPM']
    patterns = ['\\\\', '//', '--']

    labels = ['2015-12-16', '', '2015-12-18', '', '2015-12-22']
    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    plots = []
    fig = plt.figure(figsize=(3.6, 3.2))
    ax = fig.add_subplot()
    for i, s in enumerate(symbols):
        preds = data[(data['date'] >= '2015-12-16') & (data['date'] <= '2015-12-22') & (data['tick'] == s)]
        preds = preds['pred']
        plots.append(ax.bar(x - width + i * width, preds, width, label='Men', hatch=patterns[i],
                            linewidth=1, edgecolor='black'))
    ax.legend(plots, symbols)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel)
    path_out = f'../../out-fig/case-study'
    os.makedirs(path_out, exist_ok=True)
    fig.savefig(f'{path_out}/{filename}', bbox_inches='tight', dpi=300)


def main():
    prices = read_price_movements()
    run_case_study(prices, filename='prices.png', ylim=[-3.5, 3.5], ylabel='Daily price movement (%)')

    path_ours = '../out-predictions/acl18-test/0/predictions.tsv'
    data_ours = pd.read_csv(path_ours, delimiter='\t', header=None, names=['tick', 'date', 'pred'])
    run_case_study(data_ours, filename='ours.png', ylim=[-0.19, 0.19], ylabel='Predicted logit')

    path_base = '../out-baselines/Adv-ALSTM/acl18/1/predictions.tsv'
    data_base = pd.read_csv(path_base, delimiter='\t', header=None, names=['tick', 'date', 'pred'])
    run_case_study(data_base, filename='baseline.png', ylim=[-0.9, 0.9], ylabel='Predicted logit')


if __name__ == '__main__':
    main()
