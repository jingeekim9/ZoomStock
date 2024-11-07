"""
ZoomStock (BigData 2024)

Authors:
    - JinGee Kim (jingeekim9@snu.ac.kr)
    - Yong-chan Park (wjdakf3948@snu.ac.kr)
    - Jaemin Hong (jmhong0120@snu.ac.kr)
    - U Kang (ukang@snu.ac.kr)

Affiliation:
    - Data Mining Lab., Seoul National University

File: data.py
     - Functions for reading preprocessed datasets.

Version: 1.0.0
"""
import numpy as np
import os

def load_examples(data_path, trn_date, val_date, test_date, seq=2, verbose=False, with_index=False):
    """
    Load examples of a dataset.

    `with_index` should be always true for our DTML, which uses an external index as evidence.

    :param data_path: a path that contains preprocessed feature vectors.
    :param trn_date: a starting date of training data.
    :param val_date: a starting date of validation data.
    :param test_date: a starting date of test data.
    :param seq: the length of time series sequences (the same as the window parameter).
    :param verbose: whether to print the detailed process.
    :param with_index: whether to read examples with the corresponding index.
    :return: the pair of stock symbols and loaded data.
    """
    
    # Get all filenames in data_path and sort them
    fnames = sorted(os.listdir(data_path))
    fnames = [fname for fname in fnames if os.path.isfile(os.path.join(data_path, fname))]
    
    # Extract stock symbols from filenames
    ticks = [f.split('.')[0] for f in fnames]

    # Load all files as numpy arrays and store in `data`
    data = []
    for index, fname in enumerate(fnames):
        path = os.path.join(data_path, fname)
        arr = np.genfromtxt(path, dtype=float, delimiter=',', skip_header=False)
        data.append(arr)
    
    # Determine the feature dimension by subtracting 2 columns (likely date and target columns)
    x_dim = data[0].shape[1] - 2

    # If with_index is True, load the index data for reference
    if with_index:
        path = os.path.join(data_path, '..', 'index.csv')
        data_index = np.genfromtxt(path, dtype=float, delimiter=',', skip_header=False)

    # Load trading dates to map date strings to array indices
    path = os.path.join(data_path, '..', 'trading_dates.csv')
    trading_dates = np.genfromtxt(path, dtype=str, delimiter=',', skip_header=False)
    dates_idx = {date: index for index, date in enumerate(trading_dates)}

    # Define start and end indices for training, validation, and testing
    trn_idx = dates_idx[trn_date]
    val_idx = dates_idx[val_date]
    test_idx = dates_idx[test_date]

    stocks = len(fnames)
    trn_num = val_idx - max(trn_idx, seq)  # Number of training samples
    val_num = test_idx - val_idx           # Number of validation samples
    test_num = len(trading_dates) - test_idx  # Number of test samples

    # Define function to generate instances for each split
    def generate_instances(size, date_start, date_end):
        # Allocate space for the data (features and targets) and masks
        all_stocks = stocks + 1 if with_index else stocks
        out_x = np.zeros([size, all_stocks, seq, x_dim], dtype=np.float32)
        out_y = np.zeros([size, stocks], dtype=np.float32)
        out_xmask = np.zeros([size, stocks], dtype=bool)
        out_ymask = np.zeros([size, stocks], dtype=bool)

        # Populate feature and target arrays with valid data for each stock
        for tic_idx in range(stocks):
            ins_idx = 0
            for date_idx in range(date_start, date_end):
                if date_idx < seq:  # Skip if not enough history
                    continue
                # Only select instances with valid data
                if data[tic_idx][date_idx - seq:date_idx, :].min() > -123320:
                    out_x[ins_idx, tic_idx] = data[tic_idx][date_idx - seq:date_idx, :-2]
                    out_xmask[ins_idx, tic_idx] = True
                    if abs(data[tic_idx][date_idx][-2]) > 1e-8:
                        out_y[ins_idx, tic_idx] = (data[tic_idx][date_idx][-2] + 1) / 2
                        out_ymask[ins_idx, tic_idx] = True
                ins_idx += 1

        # If with_index, add the index data to the instance
        if with_index:
            tic_idx = -1
            ins_idx = 0
            for date_idx in range(date_start, date_end):
                if date_idx < seq:  # Skip if not enough history
                    continue
                if data_index[date_idx - seq:date_idx, :].min() > -123320:
                    out_x[ins_idx, tic_idx] = data_index[date_idx - seq:date_idx, :-2]
                ins_idx += 1

        # Filter instances to only those with valid target values
        survive = out_ymask.sum(1) > 0
        out_x = out_x[survive]
        out_y = out_y[survive]
        out_xmask = out_xmask[survive]
        out_ymask = out_ymask[survive]
        return out_x, out_y, out_xmask, out_ymask

    # Generate training, validation, and testing data
    trn_out = generate_instances(trn_num, trn_idx, val_idx)
    val_out = generate_instances(val_num, val_idx, test_idx)
    test_out = generate_instances(test_num, test_idx, len(trading_dates))
    data = *trn_out, *val_out, *test_out

    # Optionally print dataset details
    if verbose:
        print(len(fnames), ' tickers selected')
        print(len(trading_dates), 'trading dates:')
        print(trn_num, ' training instances')
        print(val_num, ' validation instances')
        print(test_num, ' testing instances')

    return ticks, data

def load_data(data, seq, verbose=False, with_ticks=False):
    """
    Load a dataset by the name.

    :param data: the name of a dataset to load.
    :param seq: the length of time series sequences (the same as the window parameter).
    :param verbose: whether to print the detailed process.
    :param with_ticks: whether to return the data with the list of stock symbols.
    :return: the loaded data.
    """
    # Define specific start and end dates for various datasets
    if data == 'acl18':
        tra_date = '2013-06-03'
        val_date = '2015-01-02'
        tes_date = '2015-07-01'
    elif data == 'acl23':
        tra_date = '2018-01-02'
        val_date = '2021-01-04'
        tes_date = '2022-01-04'
    elif data == 'gb':
        tra_date = '2018-01-02'
        val_date = '2021-07-01'
        tes_date = '2022-09-01'
    elif data == 'kdd23':
        tra_date = '2018-01-02'
        val_date = '2021-01-04'
        tes_date = '2022-01-04'
    elif data == 'kospi':
        tra_date = '2018-01-02'
        val_date = '2021-07-01'
        tes_date = '2022-09-01'
    elif data == 'ni225':
        tra_date = '2018-01-02'
        val_date = '2021-07-01'
        tes_date = '2022-09-01'
    elif data == 'de23':
        tra_date = '2018-01-02'
        val_date = '2021-07-01'
        tes_date = '2022-09-01'
    elif data == 'hongkong':
        tra_date = '2018-01-02'
        val_date = '2021-07-01'
        tes_date = '2022-09-01'
    elif data == 'ndx100':
        tra_date = '2013-01-02'
        val_date = '2016-01-04'
        tes_date = '2017-01-03'
    elif data == 'taiwan':
        tra_date = '2018-01-02'
        val_date = '2021-07-01'
        tes_date = '2022-09-01'
    else:
        raise ValueError(data)

    # Load dataset examples and optionally return tickers
    path = '../../../newData/data/{}/multivariate'.format(data)
    ticks, data = load_examples(path, tra_date, val_date, tes_date, seq, verbose=verbose, with_index=True)
    if with_ticks:
        return ticks, data
    else:
        return data
