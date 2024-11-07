"""
ZoomStock (BigData 2024)

Authors:
    - JinGee Kim (jingeekim9@snu.ac.kr)
    - Yong-chan Park (wjdakf3948@snu.ac.kr)
    - Jaemin Hong (jmhong0120@snu.ac.kr)
    - U Kang (ukang@snu.ac.kr)

Affiliation:
    - Data Mining Lab., Seoul National University

File: utils.py
     - Helper functions

Version: 1.0.0
"""
import torch
from torch.utils.data import DataLoader, TensorDataset

def to_device(gpu):
    """
    Determines the device (CPU or GPU) to use based on the availability of a GPU.

    :param gpu: Integer representing the GPU index to use if a GPU is available.
    :return: A torch.device object set to either the specified GPU or CPU.
    """
    # Check if GPU index is provided and if CUDA is available.
    if gpu is not None and torch.cuda.is_available():
        # Return the device pointing to the specified GPU.
        return torch.device('cuda:{}'.format(gpu))
    else:
        # Default to CPU if no GPU index is provided or CUDA is unavailable.
        return torch.device('cpu')

def to_loader(x, y, xm, ym, batch_size, shuffle=False):
    """
    Creates a DataLoader for a dataset consisting of input and output tensors.

    :param x: NumPy array for the main input data.
    :param y: NumPy array for the main target data.
    :param xm: NumPy array for auxiliary input data.
    :param ym: NumPy array for auxiliary target data.
    :param batch_size: Size of each batch.
    :param shuffle: Boolean indicating whether to shuffle the data.
    :return: DataLoader object for iterating over batches of the TensorDataset.
    """
    # Convert NumPy arrays to PyTorch tensors.
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    xm = torch.from_numpy(xm)
    ym = torch.from_numpy(ym)
    
    # Combine tensors into a TensorDataset and create a DataLoader for batching.
    return DataLoader(TensorDataset(x, y, xm, ym), batch_size, shuffle)
