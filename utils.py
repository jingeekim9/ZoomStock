import torch
from torch.utils.data import DataLoader, TensorDataset


def to_device(gpu):
    if gpu is not None and torch.cuda.is_available():
        return torch.device('cuda:{}'.format(gpu))
    else:
        return torch.device('cpu')


def to_loader(x, y, xm, ym, batch_size, shuffle=False):
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    xm = torch.from_numpy(xm)
    ym = torch.from_numpy(ym)
    return DataLoader(TensorDataset(x, y, xm, ym), batch_size, shuffle)
