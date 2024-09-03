import argparse
import os

import pandas as pd

import numpy as np
import torch
from torch import nn

from data import load_data, get_date_ranges
from models import DTML
from utils import to_device, to_loader
from matplotlib import pyplot as plt


class Ensemble(nn.Module):
    def __init__(self, args, in_features, num_stocks, seeds):
        super().__init__()
        models = []
        for seed in seeds:
            model_path = os.path.join(args.out, args.data, str(seed), 'model.pth')
            model = DTML(in_features, num_stocks, args.units, beta=args.beta)
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            models.append(model)
        self.models = nn.ModuleList(models)

    def forward(self, *inputs):
        out_list = []
        for model in self.models:
            out_list.append(torch.sigmoid(model(*inputs)))
        return torch.stack(out_list).mean(dim=0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='acl18')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=str, default='../out')
    parser.add_argument('--window', type=int, default=10)
    parser.add_argument('--device', type=int, default=None)
    parser.add_argument('--units', type=int, default=64)
    parser.add_argument('--beta', type=float, default=1)
    return parser.parse_args()


def read_closing_prices(args, ticks):
    data_path = f'../data/{args.data}/ourpped'
    path = os.path.join(data_path, '..', 'trading_dates.csv')
    dates = np.genfromtxt(path, dtype=str, delimiter=',', skip_header=False)
    data = []
    for index, tick in enumerate(ticks):
        path = os.path.join(data_path, tick + '.csv')
        arr = np.genfromtxt(path, dtype=float, delimiter=',', skip_header=False)
        data.append(arr[:, -1])
    data = np.array(data).transpose()
    return pd.DataFrame(data, columns=ticks, index=dates)


def invest_by_index(index_name, dates):
    index_data = pd.read_csv(f'../data/index/{index_name}/raw.csv', index_col='Date')
    index_data = index_data.loc[dates, 'Adj Close']
    return index_data / index_data[0]


def invest_by_baselines(data, prices, dates, top_k):
    money_list_all = []
    for seed in range(10):
        try:
            df = pd.read_csv(f'../out-baselines/Adv-ALSTM/{data}/{seed}/predictions.tsv',
                             delimiter='\t', header=None, names=['tick', 'date', 'pred'])
        except FileNotFoundError:
            continue
        money = 1
        money_list = []
        prev_values = []
        for idx in range(len(dates)):
            date = dates[idx]
            if idx > 0:
                y_pred = df[df['date'] == date]
                y_pred = y_pred.sort_values(by='pred', ascending=False).reset_index(drop=True)
                portfolio = y_pred['tick'][:top_k].values
                money *= (prices.loc[date, portfolio] / prev_values[portfolio]).mean()
            prev_values = prices.loc[date, :]
            money_list.append(money)
        money_list_all.append(money_list)
    return money_list_all


def get_sharp_ratio(values_list, leap=10):
    ratios = []
    for values in values_list:
        profits = []
        for idx in range(leap, len(values), leap):
            profit_ours = (values[idx] - values[idx - leap]) / values[idx - leap]
            profits.append(profit_ours - 0.00099)
        ratios.append(np.mean(profits) / np.std(profits))
    print('{:.4f}\t{:.4f}'.format(np.mean(ratios), np.std(ratios)))


def invest_by_average(prices, dates):
    money = 1
    money_list = []
    p_values = []
    for idx in range(len(dates)):
        date = dates[idx]
        if idx > 0:
            money *= (prices.loc[date, :] / p_values).mean()
        p_values = prices.loc[date, :]
        money_list.append(money)
    return money_list


def do_investment(args, top_k=5, ensemble=False, with_std=False):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    _, _, date_start, date_end = get_date_ranges(args.data)
    ticks, data = load_data(args.data, args.window, with_ticks=True)
    trn_x, trn_y, trn_xm, trn_ym, val_x, val_y, val_xm, val_ym, test_x, test_y, test_xm, test_ym = data

    seeds = range(10)
    in_features = trn_x.shape[3]
    num_stocks = trn_x.shape[1] - 1
    device = to_device(args.device)

    models = []
    if ensemble:
        model = Ensemble(args, in_features, num_stocks, seeds)
        model = model.to(device)
        model.eval()
        models.append(model)
    else:
        for seed in seeds:
            model_path = os.path.join(args.out, args.data, str(seed), 'model.pth')
            model = DTML(in_features, num_stocks, args.units, beta=args.beta)
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model = model.to(device)
            model.eval()
            models.append(model)

    test_loader = to_loader(test_x, test_y, test_xm, test_ym, batch_size=1)

    path = f'../data/{args.data}/trading_dates.csv'
    dates = np.genfromtxt(path, dtype=str, delimiter=',', skip_header=False)
    dates = dates[list(dates).index(date_start):list(dates).index(date_end)]
    prices = read_closing_prices(args, ticks)

    # for seed, model in enumerate(models):
    #     predictions = []
    #     for idx, (x, y, xm, ym) in enumerate(test_loader):
    #         assert False not in xm
    #         date = dates[idx]
    #         y_pred = model(x.to(device), xm.to(device)).view(-1)
    #         for j in range(y_pred.size(0)):
    #             predictions.append((ticks[j], date, y_pred[j].item()))
    #     df = pd.DataFrame(predictions)
    #     os.makedirs(f'../out-cases/tmp/{seed}', exist_ok=True)
    #     df.to_csv(f'../out-cases/tmp/{seed}/predictions.tsv', header=False, index=False, sep='\t')

    ours_all = []
    for model in models:
        money = 1
        money_list = []
        prev_values = []
        for idx, (x, y, xm, ym) in enumerate(test_loader):
            assert False not in xm
            date = dates[idx]
            if idx > 0:
                y_pred = model(x.to(device), xm.to(device)).view(-1)
                portfolio = [ticks[i] for i in torch.topk(y_pred, k=top_k)[1]]
                diff = prices.loc[date, portfolio] / prev_values[portfolio]
                money *= diff.mean()
            prev_values = prices.loc[date, :]
            money_list.append(money)

        ours_all.append(money_list)
    ours_avg = np.stack(ours_all).mean(axis=0)
    ours_std = np.stack(ours_all).std(axis=0)

    index_name = args.data
    if args.data in ['acl18', 'kdd17']:
        index_name = 'snp500'
    index = invest_by_index(index_name, dates)
    all_stocks = invest_by_average(prices, dates)
    best_competitor = max(index[-1], all_stocks[-1])

    base_avg = None
    if args.data in ['acl18', 'kdd17', 'csi300', 'ndx100']:
        base_all = invest_by_baselines(args.data, prices, dates, top_k)
        base_avg = np.stack(base_all).mean(axis=0)
        for p in base_avg:
            print(p)
        best_competitor = max(best_competitor, base_avg[-1])

    print('{}\t{}\t{}\t{}\t{}\t{}'.format(
        args.data, ensemble, top_k, ours_avg[-1], best_competitor, ours_avg[-1] - best_competitor))

    n_data = len(ours_avg)
    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_subplot()
    ax.set_ylabel('Portfolio value')
    ax.set_xticks([0, (n_data - 1) // 3, (n_data - 1) * 2 // 3, n_data - 1])

    l1, = ax.plot(ours_avg)
    if with_std and not ensemble:
        ax.fill_between(index.index, ours_avg - ours_std, ours_avg + ours_std, alpha=0.5)
    if base_avg is not None:
        l2, = ax.plot(base_avg)
    l3, = ax.plot(all_stocks)
    l4, = ax.plot(index)
    if base_avg is not None:
        ax.legend([l1, l2, l3, l4], ['DTML (ours)', 'Adv-ALSTM', 'All Stocks', index_name.upper()],
                  loc='upper center', ncol=4, bbox_to_anchor=(0.45, 1.18))
    else:
        ax.legend([l1, l3, l4], ['DTML (ours)', 'All Stocks', 'Index'],
                  loc='upper center', ncol=3, bbox_to_anchor=(0.45, 1.18))

    path_out = f'../out-fig/investment/top-{top_k}/{args.data}'
    file_out = 'ens.png' if ensemble else f'{args.seed}.png'
    os.makedirs(path_out, exist_ok=True)
    fig.savefig(f'{path_out}/{file_out}', bbox_inches='tight', dpi=300)


def main():
    args = parse_args()
    if args.data == 'acl18':
        args.units = 64
        args.beta = 0.1
        args.window = 10
    elif args.data == 'kdd17':
        args.units = 64
        args.beta = 0.01
        args.window = 15
    elif args.data == 'ndx100':
        args.units = 128
        args.beta = 0.01
        args.window = 15
    elif args.data == 'csi300':
        args.units = 64
        args.beta = 1
        args.window = 15
    else:
        raise ValueError()
    do_investment(args, top_k=3, ensemble=False)


if __name__ == '__main__':
    main()
