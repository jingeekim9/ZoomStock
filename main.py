"""
Main function to train our DTML model and to evaluate it.
"""
import argparse
import io
import os
import shutil
import time

import torch
from torch import optim, nn
import numpy as np

import models
from data import load_data
# from eval.invest import read_closing_prices
from utils import to_device, to_loader


def to_conf_matrix(predictions, labels, mask):
    """
    Make a confusion matrix from predictions and true labels.

    The `mask` argument is given because not all examples are used in our training and evaluation;
    only the ones whose movement ratios are larger than a certain threshold are used for both
    training and evaluation, as done in previous works.

    :param predictions: a list of model predictions.
    :param labels: a list of true labels.
    :param mask: a mask indicating valid examples.
    :return: the generated confusion matrix.
    """
    pos_samples = labels[(predictions >= 0) & mask]
    neg_samples = labels[(predictions < 0) & mask]
    tp = (pos_samples == 1).sum().cpu()
    fp = (pos_samples == 0).sum().cpu()
    fn = (neg_samples == 1).sum().cpu()
    tn = (neg_samples == 0).sum().cpu()
    return np.array([[tp, fp], [fn, tn]], dtype=np.int64)


def to_acc(conf_matrix):
    """
    Calculate the accuracy from a confusion matrix.

    :param conf_matrix: a confusion matrix.
    :return: the accuracy.
    """
    tp = conf_matrix[0, 0]
    tn = conf_matrix[1, 1]
    return (tp + tn) / conf_matrix.sum()


def to_mcc(conf_matrix):
    """
    Calculate the Matthew's correlation coefficient (MCC) from a confusion matrix.

    This metric is similar to accuracy, but works well with unbalanced datasets unlike accuracy.
    Refer to the paper for the exact definition of MCC.

    :param conf_matrix: a confusion matrix.
    :return: the MCC.
    """
    conf_matrix = conf_matrix.astype(dtype=np.float32)
    tp = conf_matrix[0, 0]
    fp = conf_matrix[0, 1]
    fn = conf_matrix[1, 0]
    tn = conf_matrix[1, 1]
    if min(tp + fp, tp + fn, tn + fp, tn + fn) == 0:
        return 0
    return (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))


class Trainer:
    """
    A class for training and evaluation our model.
    """
    def __init__(self, model, optimizer, device, decay=0):
        """
        Initializer function.

        :param model: a PyTorch module of DTML.
        :param optimizer: an optimizer for the model.
        :param device: a device where the model is stored.
        :param decay: the decaying parameter for the last predictor of DTML.
        """
        self.decay = decay
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.loss = nn.BCEWithLogitsLoss(reduction='none')

    def train(self, loader):
        """
        Train the model by a single epoch given a data loader.

        :param loader: a data loader for training.
        :return: the training loss.
        """
        self.model.train()
        loss_sum, count = 0, 0
        for x, y, xm, ym in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            xm = xm.to(self.device)
            ym = ym.to(self.device)
            y_pred = self.model(x, xm)
            loss = self.loss(y_pred, y.float())[ym].mean() + self.decay * self.model.l2_loss()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_sum += loss.item() * ym.sum()
            count += ym.sum()
        return loss_sum / count

    def evaluate(self, loader):
        """
        Evaluate the model given a data loader.

        :param loader: a data loader for evaluation.
        :return: the accuracy (ACC) and MCC.
        """
        self.model.eval()
        conf_matrix = np.zeros((2, 2), dtype=np.int64)
        for x, y, xm, ym in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            xm = xm.to(self.device)
            ym = ym.to(self.device)
            y_pred = self.model(x, xm)
            conf_matrix += to_conf_matrix(y_pred, y, ym)
        return to_acc(conf_matrix), to_mcc(conf_matrix)


def evaluate_investment(args, model, device, top_k):
    """
    Simulate an actual investment and report the return.

    This functions needs not to be run for new datasets; do not set the `invest` argument of the
    main function to True.

    :param args: an argument parser.
    :param model: a trained DTML model.
    :param device: a device where the model is stored.
    :param top_k: the number of stocks to invest at each day.
    :return: the return of the investment.
    """
    if args.data == 'acl18':
        date_start = '2015-10-01'
    elif args.data == 'kdd17':
        date_start = '2016-01-04'
    elif args.data == 'kospi':
        date_start = '2018-01-02'
    else:
        raise ValueError(args.data)

    ticks, data = load_data(args.data, args.window, with_ticks=True)
    test_data = data[-4:]
    test_loader = to_loader(*test_data, batch_size=128)

    path = f'../data/{args.data}/trading_dates.csv'
    dates = np.genfromtxt(path, dtype=str, delimiter=',', skip_header=False)
    dates = dates[list(dates).index(date_start):]
    prices = read_closing_prices(args, ticks)

    predictions = []
    for idx, (x, y, xm, ym) in enumerate(test_loader):
        assert False not in xm
        x = x.to(device)
        xm = xm.to(device)
        y_pred = model(x, xm)
        predictions.append(y_pred)
    predictions = torch.cat(predictions)

    money = 1
    money_list = []
    prev_values = []
    for idx, pred in enumerate(predictions):
        date = dates[idx]
        if idx > 0:
            portfolio = [ticks[i] for i in torch.topk(pred, k=top_k)[1]]
            diff = prices.loc[date, portfolio] / prev_values[portfolio]
            money *= diff.mean()
        prev_values = prices.loc[date, :]
        money_list.append(money)
    return money_list[-1]


def parse_args():
    """
    Parse command line arguments for the script.

    :return: the parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='acl18')
    parser.add_argument('--seed', type=int, default=0)

    #
    # Major hyperparameters that are very important for the performance.
    #
    # `beta` determines how much we accept the global index (Eq (5)).
    parser.add_argument('--beta', type=float, default=1)  # [0.01, 0.1, 1]
    # `decay` is used to prevent overfitting.
    parser.add_argument('--decay', type=float, default=1)  # [0.01, 0.1, 1, 10]
    # `window` is the length of time series inputs.
    parser.add_argument('--window', type=int, default=100)  # [10, 15]

    #
    # Minor hyperparameters that are still important for the performance.
    #
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--units', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)

    #
    # Minor hyperparameters that typically have little effects.
    #
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--patience', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--attn-heads', type=int, default=4)
    parser.add_argument('--attn-layers', type=int, default=1)
    parser.add_argument('--mlp-factor', type=int, default=4)

    #
    # Environment variables that need not be modified during experiments.
    #
    parser.add_argument('--invest', action='store_true', default=False)
    parser.add_argument('--load', action='store_true', default=False)
    parser.add_argument('--no-stop', action='store_true', default=False)
    parser.add_argument('--out', type=str, default='../out')
    parser.add_argument('--device', type=int, default=None)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--silent', action='store_true', default=False)
    return parser.parse_args()


def main():
    """
    Main function for training and evaluation.

    :return: None.
    """
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    trn_x, trn_y, trn_xm, trn_ym, val_x, val_y, val_xm, val_ym, test_x, test_y, test_xm, test_ym = \
        load_data(args.data, args.window)
    in_features = trn_x.shape[3]  # examples x variables x window x features
    num_stocks = trn_x.shape[1] - 1

    model = models.DTML(in_features,
                        num_stocks=num_stocks,
                        hidden_size=args.units,
                        beta=args.beta,
                        dropout=args.dropout,
                        attn_heads=args.attn_heads,
                        attn_layers=args.attn_layers,
                        mlp_factor=args.mlp_factor,
                        window=args.window)

    device = to_device(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    trainer = Trainer(model, optimizer, device, decay=args.decay)

    trn_loader = to_loader(trn_x, trn_y, trn_xm, trn_ym, args.batch_size, shuffle=True)
    val_loader = to_loader(val_x, val_y, val_xm, val_ym, args.batch_size)
    test_loader = to_loader(test_x, test_y, test_xm, test_ym, args.batch_size)

    out_path = os.path.join(args.out, str(args.seed))

    if args.load:
        save_path = os.path.join(out_path, 'model.pth')
        model.load_state_dict(torch.load(save_path))
    else:
        if os.path.exists(out_path):
            shutil.rmtree(out_path)
        os.makedirs(out_path, exist_ok=True)

        saved_model, best_epoch, best_acc = io.BytesIO(), 0, -np.inf
        start_time = time.time()
        for epoch in range(args.epochs + 1):
            trn_loss = 0
            if epoch > 0:
                trn_loss = trainer.train(trn_loader)
            val_acc, val_mcc = trainer.evaluate(val_loader)
            test_acc, test_mcc = trainer.evaluate(test_loader)
            log = '{:3d}\t{:7.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(
                epoch, trn_loss, val_acc, val_mcc, test_acc, test_mcc)
            if epoch >= 10 and val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                saved_model.seek(0)
                torch.save(model.state_dict(), saved_model)
                log += '\tBEST'
            with open(os.path.join(out_path, 'log.tsv'), 'a') as f:
                f.write(log + '\n')
            if not args.silent:
                print(log)

            if args.patience > 0 and epoch >= best_epoch + args.patience:
                break
        end_time = time.time()
        
        train_time = end_time - start_time
        print("Training time: ", train_time, "seconds")

        if not args.no_stop:
            saved_model.seek(0)
            model.load_state_dict(torch.load(saved_model))

    trn_res = trainer.evaluate(trn_loader)
    val_res = trainer.evaluate(val_loader)
    test_res = trainer.evaluate(test_loader)
    log = '{}\t{}\t{}\t{}\t{}\t{}'.format(*trn_res, *val_res, *test_res)

    if args.invest and args.data in ['acl18', 'kdd17', 'jpn20', 'gb20', 'chn20', 'crypto']:
        test_invest = []
        for k in [3, 5, 10, 20]:
            test_invest.append(evaluate_investment(args, model, device, top_k=k))
        log += '\t{}\t{}\t{}\t{}'.format(*test_invest)

    if not args.silent:
        print(log)

    with open(os.path.join(out_path, 'out.tsv'), 'w') as f:
        f.write(log + '\n')

    if args.save:
        save_path = os.path.join(out_path, 'model.pth')
        torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    main()
