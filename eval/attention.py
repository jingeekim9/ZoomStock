import argparse
import os
import numpy as np
import torch

from data import load_data
from models import DTML
from utils import to_device, to_loader
from matplotlib import pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=str, default='../out')
    parser.add_argument('--device', type=int, default=None)

    # Hyperparameters that should not be changed.
    parser.add_argument('--data', type=str, default='acl18')
    parser.add_argument('--window', type=int, default=10)
    parser.add_argument('--units', type=int, default=64)
    parser.add_argument('--beta', type=float, default=0.1)
    return parser.parse_args()


def get_attention_maps(args, mode='spatial'):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = to_device(args.device)

    test_x, test_y, test_xm, test_ym = load_data(args.data, args.window)[-4:]
    test_loader = to_loader(test_x, test_y, test_xm, test_ym, batch_size=1)
    in_features = test_x.shape[3]
    num_stocks = test_x.shape[1] - 1

    model_path = os.path.join(args.out, args.data, str(args.seed), 'model.pth')
    model = DTML(in_features, num_stocks, args.units, args.beta)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = model.to(device)
    model.eval()

    data_path = f'../data/{args.data}'
    dates = np.genfromtxt(
        os.path.join(data_path, 'trading_dates.csv'), dtype=str, delimiter=',', skip_header=False)
    dates = dates[list(dates).index('2015-10-01'):]

    symbols = [fname.split('.')[0]
               for fname in os.listdir(os.path.join(data_path, 'ourpped'))
               if os.path.isfile(os.path.join(data_path, 'ourpped', fname))]

    attn_maps = []
    for idx, (x, y, xm, ym) in enumerate(test_loader):
        x = x.to(device)
        y_pred, t_attn, s_attn = model(x, xm, with_attn=True)
        if mode.startswith('spatial'):
            attn_map = s_attn[0].squeeze(0)
        elif mode == 'temporal':
            attn_map = t_attn
        else:
            raise ValueError(mode)
        attn_maps.append(attn_map.detach().numpy())

    return symbols, dates, attn_maps


# This function is not used currently.
def all_pair_similarities(attn_map, symmetric=False):
    def kl_div(a, b):
        return (a * (a.log2() - b.log2())).sum(dim=2)

    attn_map = torch.from_numpy(attn_map)
    if symmetric:
        a1 = attn_map.unsqueeze(1)
        a2 = attn_map.unsqueeze(0)
        m = (a1 + a2) / 2
        v1 = kl_div(a1, m)
        v2 = kl_div(a2, m)
        dist = (v1 + v2) / 2
    else:
        a1 = attn_map.unsqueeze(1)
        a2 = attn_map.unsqueeze(0)
        dist = kl_div(a1, a2)
    return 1 - dist


def visualize_attention(args, target, sort=True):
    out_path = '../out-fig'
    if target == 'spatial-1':
        symbols, dates, attn_maps = get_attention_maps(args, target)
        for i in range(len(dates)):
            attn_map = attn_maps[i]
            if sort:
                scores = attn_map.sum(axis=0)
                scores = sorted(list(enumerate(scores)), key=lambda e: e[1], reverse=True)
                stocks = [e[0] for e in scores]
            else:
                stocks = list(range(len(symbols)))

            date = ''.join(dates[i].split('-'))

            # Print the scores by source stocks.
            out_file = '{}/stock-attention/scores-by-sources/{}.tsv'.format(out_path, date)
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            with open(out_file, 'w') as f:
                pairs = list(zip(enumerate(symbols), attn_map.sum(axis=0)))
                pairs = sorted(pairs, key=lambda e: e[1], reverse=True)
                for p in pairs:
                    f.write('{}\t{}\t{}\n'.format(p[0][0], p[0][1], p[1]))

            # Write the scores sorted by individual pairs.
            out_file = '{}/stock-attention/scores-by-pairs/{}.tsv'.format(out_path, date)
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            with open(out_file, 'w') as f:
                tuples = []
                for j in range(len(symbols)):
                    for k in range(len(symbols)):
                        tuples.append(((k, symbols[k]), (j, symbols[j]), attn_map[j, k]))
                tuples = sorted(tuples, key=lambda e: e[-1], reverse=True)
                tuples = [t for t in tuples if t[0][0] != t[1][0]]
                for t in tuples:
                    if t[1][1] == 'FB':
                        # if t[0][1] in ['AAPL', 'GOOG', 'AMZN']:
                        f.write('{}\t{}\t{}\t{}\t{:.4f}\n'.format(
                            t[0][0], t[0][1], t[1][0], t[1][1], t[2]))

            # Reorder the stocks.
            new_attn_map = np.zeros_like(attn_map)
            for j in range(len(symbols)):
                for k in range(len(symbols)):
                    new_attn_map[j, k] = attn_map[stocks[j], stocks[k]]
            attn_map = new_attn_map

            out_file = '{}/stock-attention/figures/{}.png'.format(out_path, date)
            os.makedirs(os.path.dirname(out_file), exist_ok=True)

            plt.rc('font', size=18)
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.set_xlabel('Source stock index')
            ax.set_ylabel('Target stock index')
            im = ax.imshow(attn_map)
            fig.colorbar(im, orientation='horizontal', pad=-1.65)
            fig.savefig(out_file, dpi=300, bbox_inches='tight')
            plt.close()

    if target == 'spatial-2':
        symbols, dates, attn_maps = get_attention_maps(args, mode='spatial')

        for symbol in ['AMZN', 'GOOG']:
            idx = list(symbols).index(symbol)
            attn_map = np.array([attn[idx, :] for attn in attn_maps]).transpose()

            if symbol == 'GOOG':
                max_scores = attn_map.max(axis=1)
                scores = sorted(list(zip(list(range(len(symbols))), symbols, max_scores)),
                                key=lambda x: x[2], reverse=True)
                for s in scores[:10]:
                    print(s)

            out_file = '{}/stock-attention/stocks/{}.png'.format(out_path, symbol)
            os.makedirs(os.path.dirname(out_file), exist_ok=True)

            plt.rc('font', size=18)
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.set_xlabel('Date (2015)')
            ax.set_xticks([0, 22, 42, 63])
            ax.set_xticklabels(['10-01', '11-02', '12-01', '12-31'])
            if symbol == 'AMZN':
                ax.set_yticks([0, 79, 85, 6, 16])
                ax.set_yticklabels(['CSCO', 'AAPL', 'REX', 'TM', 'SNP'])
            elif symbol == 'GOOG':
                ax.set_yticks([16, 46, 35, 6, 56])
                ax.set_yticklabels(['SNP', 'RDS-B', 'MMM', 'TM', 'DIS'])

            im = ax.imshow(attn_map)
            fig.colorbar(im, orientation='horizontal', pad=-2.1)
            fig.savefig(out_file, dpi=300, bbox_inches='tight')
            plt.close()

    elif target == 'temporal':
        symbols, dates, attn_maps = get_attention_maps(args, target)

        for target_idx, target in enumerate(symbols):
            attn_map = np.full((len(dates), len(dates)), fill_value=np.nan)
            for i in range(len(dates)):
                attn_map[i, max(i - 9, 0):i + 1] = attn_maps[i][target_idx][-(i + 1):]

            out_file = '{}/temporal-attention/figures/{}.png'.format(out_path, target)
            os.makedirs(os.path.dirname(out_file), exist_ok=True)

            plt.rc('font', size=14)
            plt.xticks([0, 22, 42, 63], ['10-01', '11-02', '12-01', '12-31'])
            plt.xlabel('Source date (2015)')
            y_idx = [0, 10, 20, 30, 40, 50, 60]
            plt.yticks(y_idx, [d[-5:] for d in dates[y_idx]])
            plt.ylabel('Target date (2015)')
            plt.imshow(attn_map)
            plt.colorbar()
            plt.savefig(out_file, dpi=300, bbox_inches='tight')
            plt.close()

    else:
        raise ValueError(target)


def main():
    args = parse_args()
    assert args.data == 'acl18'
    if args.target == 'spatial-1':
        visualize_attention(args, args.target, sort=True)
    elif args.target == 'spatial-2':
        visualize_attention(args, args.target)
    elif args.target == 'temporal':
        visualize_attention(args, args.target)
    else:
        raise ValueError(args.target)


if __name__ == '__main__':
    main()
