"""
ZoomStock (BigData 2024)

Authors:
    - JinGee Kim (jingeekim9@snu.ac.kr)
    - Yong-chan Park (wjdakf3948@snu.ac.kr)
    - Jaemin Hong (jmhong0120@snu.ac.kr)
    - U Kang (ukang@snu.ac.kr)

Affiliation:
    - Data Mining Lab., Seoul National University

File: run.py
     - Run main.py N times with different random seeds and GPUs.

Version: 1.0.0
"""

import argparse
import multiprocessing
import os

import numpy as np
import subprocess
from multiprocessing import Pool


def parse_args():
    """
    Parse command line arguments.

    The other arguments not defined in this function are directly passed to main.py. For instance,
    an option like "--beta 1" is given directly to the main script.

    :return: the parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--out', type=str, default='../dtml_multivariate_2')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(5)))
    parser.add_argument('--gpus', type=int, nargs='+', default=0)
    parser.add_argument('--workers', type=int, default=1)
    return parser.parse_known_args()


def run_command(args):
    """
    Run main.py with a suitable GPU given as an argument.

    :param args: the pair of a command and a list of GPUs.
    :return: None.
    """
    command, gpu_list = args
    gpu_idx = int(multiprocessing.current_process().name.split('-')[-1]) - 1
    gpu = 0
    command += ['--device', str(gpu)]
    subprocess.call(command)


def main():
    """
    Main function to run main.py with multiple jobs.

    :return: None.
    """
    args, unknown = parse_args()
    assert args.data is not None

    out_path = '{}/{}'.format(args.out, args.data)
    args_list = []
    for seed in args.seeds:
        command = ['python', 'main.py',
                   '--data', args.data,
                   '--seed', str(seed),
                   '--out', out_path]
        args_list.append((command + unknown, args.gpus))

    with Pool(1 * args.workers) as pool:
        pool.map(run_command, args_list)

    values = []
    for seed in args.seeds:
        values.append(np.loadtxt(os.path.join(out_path, str(seed), 'out.tsv'), delimiter='\t'))
    avg = np.stack(values, axis=0).mean(axis=0)
    std = np.stack(values, axis=0).std(axis=0)
    for a, s in zip(avg, std):
        print('{:.4f}\t{:.4f}'.format(a, s), end='\t')
    print()


if __name__ == '__main__':
    main()
