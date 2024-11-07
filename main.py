"""
ZoomStock (BigData 2024)

Authors:
    - JinGee Kim (jingeekim9@snu.ac.kr)
    - Yong-chan Park (wjdakf3948@snu.ac.kr)
    - Jaemin Hong (jmhong0120@snu.ac.kr)
    - U Kang (ukang@snu.ac.kr)

Affiliation:
    - Data Mining Lab., Seoul National University

File: main.py
     - Main function to train our ZoomStock model and to evaluate it.

Version: 1.0.0
"""

import argparse  # For parsing command-line arguments
import io       # For input/output operations
import os       # For operating system interfaces
import shutil   # For file operations
import time     # For timing the training process

import torch  # PyTorch library for deep learning
from torch import optim, nn  # Optimizer and neural network module
import numpy as np  # Numerical operations

import models  # Custom module where model classes are defined
from data import load_data  # Data loading function from custom data module
from utils import to_device, to_loader  # Utility functions for device management and data loaders


def to_conf_matrix(predictions, labels, mask):
    """
    Create a confusion matrix from predictions and true labels.

    :param predictions: Tensor of model predictions.
    :param labels: Tensor of true labels.
    :param mask: Mask for filtering out examples below a threshold.
    :return: Confusion matrix in a 2x2 numpy array.
    """
    pos_samples = labels[(predictions >= 0) & mask]  # True positives & false positives
    neg_samples = labels[(predictions < 0) & mask]   # True negatives & false negatives
    tp = (pos_samples == 1).sum().cpu()  # True positives
    fp = (pos_samples == 0).sum().cpu()  # False positives
    fn = (neg_samples == 1).sum().cpu()  # False negatives
    tn = (neg_samples == 0).sum().cpu()  # True negatives
    return np.array([[tp, fp], [fn, tn]], dtype=np.int64)


def to_acc(conf_matrix):
    """
    Calculate accuracy from a confusion matrix.

    :param conf_matrix: Confusion matrix.
    :return: Accuracy as a float.
    """
    tp = conf_matrix[0, 0]
    tn = conf_matrix[1, 1]
    return (tp + tn) / conf_matrix.sum()  # Sum of true positives and true negatives / total


def to_mcc(conf_matrix):
    """
    Calculate the Matthews correlation coefficient (MCC) from a confusion matrix.

    :param conf_matrix: Confusion matrix.
    :return: MCC value as a float.
    """
    conf_matrix = conf_matrix.astype(dtype=np.float32)
    tp, fp = conf_matrix[0, 0], conf_matrix[0, 1]
    fn, tn = conf_matrix[1, 0], conf_matrix[1, 1]
    if min(tp + fp, tp + fn, tn + fp, tn + fn) == 0:
        return 0  # Avoid division by zero in case of empty classes
    return (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))


class Trainer:
    """
    Trainer class for handling training and evaluation of the model.
    """
    def __init__(self, model, optimizer, device, decay=0):
        """
        Initializes the Trainer class.

        :param model: PyTorch model (ZoomStock).
        :param optimizer: Optimizer for the model.
        :param device: Device where model is stored (CPU/GPU).
        :param decay: Regularization parameter for model.
        """
        self.decay = decay
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.loss = nn.BCEWithLogitsLoss(reduction='none')  # Loss for binary classification

    def train(self, loader):
        """
        Train model for a single epoch.

        :param loader: DataLoader for training data.
        :return: Average training loss.
        """
        self.model.train()  # Set model to training mode
        loss_sum, count = 0, 0
        for x, y, xm, ym in loader:
            # Move data to specified device
            x, y, xm, ym = x.to(self.device), y.to(self.device), xm.to(self.device), ym.to(self.device)
            y_pred = self.model(x, xm)  # Predict
            loss = self.loss(y_pred, y.float())[ym].mean() + self.decay * self.model.l2_loss()
            self.optimizer.zero_grad()  # Reset gradients
            loss.backward()  # Backpropagate loss
            self.optimizer.step()  # Update weights
            loss_sum += loss.item() * ym.sum()  # Update loss sum
            count += ym.sum()  # Count valid samples
        return loss_sum / count

    def evaluate(self, loader):
        """
        Evaluate the model.

        :param loader: DataLoader for evaluation data.
        :return: Accuracy and MCC.
        """
        self.model.eval()  # Set model to evaluation mode
        conf_matrix = np.zeros((2, 2), dtype=np.int64)
        for x, y, xm, ym in loader:
            x, y, xm, ym = x.to(self.device), y.to(self.device), xm.to(self.device), ym.to(self.device)
            y_pred = self.model(x, xm)  # Predict
            conf_matrix += to_conf_matrix(y_pred, y, ym)  # Update confusion matrix
        return to_acc(conf_matrix), to_mcc(conf_matrix)


def evaluate_investment(args, model, device, top_k):
    """
    Simulate an investment strategy.

    :param args: Argument parser.
    :param model: Trained model.
    :param device: Device to run the model.
    :param top_k: Top stocks to select daily.
    :return: Final return from investment.
    """
    # Define start date based on dataset
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

    dates = np.genfromtxt(f'../data/{args.data}/trading_dates.csv', dtype=str, delimiter=',', skip_header=False)
    dates = dates[list(dates).index(date_start):]
    prices = read_closing_prices(args, ticks)

    # Generate predictions
    predictions = []
    for idx, (x, y, xm, ym) in enumerate(test_loader):
        assert False not in xm
        x, xm = x.to(device), xm.to(device)
        predictions.append(model(x, xm))
    predictions = torch.cat(predictions)

    # Investment simulation
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
    Parse command-line arguments.

    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='acl18')
    parser.add_argument('--seed', type=int, default=0)
    # Major hyperparameters
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--decay', type=float, default=1)
    parser.add_argument('--window', type=int, default=100)
    # Minor hyperparameters
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--units', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    # Other parameters
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--patience', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--attn-heads', type=int, default=4)
    parser.add_argument('--attn-layers', type=int, default=1)
    parser.add_argument('--mlp-factor', type=int, default=4)
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
    Main function for training and evaluating the ZoomStock model.

    :return: None.
    """
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    trn_x, trn_y, trn_xm, trn_ym, val_x, val_y, val_xm, val_ym, test_x, test_y, test_xm, test_ym = load_data(args.data, args.window)
    trn_loader = to_loader(trn_x, trn_y, trn_xm, trn_ym, batch_size=args.batch_size)
    val_loader = to_loader(val_x, val_y, val_xm, val_ym, batch_size=args.batch_size)
    test_loader = to_loader(test_x, test_y, test_xm, test_ym, batch_size=args.batch_size)

    device = torch.device(f'cuda:{args.device}' if args.device else 'cpu')
    model = models.ZoomStock(args.units, args.beta, args.window, args.attn_heads, args.attn_layers, args.mlp_factor, args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    trainer = Trainer(model, optimizer, device, decay=args.decay)

    # Training and evaluation loop
    best_mcc, best_epoch = 0, 0
    for epoch in range(args.epochs):
        trainer.train(trn_loader)  # Training step
        _, mcc = trainer.evaluate(val_loader)  # Validation step

        if mcc > best_mcc:
            best_mcc, best_epoch = mcc, epoch
            if args.save:
                torch.save(model.state_dict(), os.path.join(args.out, 'model.pt'))

        # Early stopping condition
        if args.patience and epoch - best_epoch >= args.patience:
            break

    if args.load:
        model.load_state_dict(torch.load(os.path.join(args.out, 'model.pt')))
    acc, mcc = trainer.evaluate(test_loader)

    # Investment evaluation if specified
    if args.invest:
        investment_return = evaluate_investment(args, model, device, top_k=10)
        print(f'Investment Return: {investment_return:.2f}')

    print(f'Accuracy: {acc:.4f}, MCC: {mcc:.4f}')
