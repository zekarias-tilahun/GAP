import numpy as np
import argparse
import random
import sys

VERBOSE = False
ERROR = 'ERROR'
INFO = 'INFO'
PROG = 'PROG'


def read_embedding(path):
    log('Reading embedding from {}'.format(path))
    embeddings = {}
    with open(path, 'rb') as f:
        for line in f:
            ln = line.strip().split()
            node = int(ln[0])
            embeddings[node] = list(map(float, ln[1:]))
    return embeddings


def log(msg, ret=False, progress=None, intervals=None, level=INFO):
    global VERBOSE
    if VERBOSE:
        if ret or progress is not None:
            if progress is None:
                sys.stdout.write(f'\r{level}: {msg}')
            else:
                #sys.stdout.write(f'\rPROG: Batch {progress}/{len(intervals)}')
                p = f"{PROG}: {msg} [" + ''.join('=' for _ in range(progress))
                space = ' '.join('' for _ in range(100 - progress)) + f'{np.searchsorted(intervals, progress)}%]'
                sys.stdout.write(f'\r{np.searchsorted(intervals, progress)}%')
            sys.stdout.flush()
        else:
            print(f"INFO: {msg}")


def scale_min_max(old_values, new_max, new_min):
    old_min = old_values.min()
    old_max = old_values.max()
    val_std = (old_values - old_min) / (old_max - old_min)
    return val_std * (new_max - new_min) + new_min


def visualize_attention(nodes, weights):
    weights_ = scale_min_max(weights[weights != 0].data.numpy(), new_max=1, new_min=0)
    nodes_ = nodes[nodes != 0]
    with open('attention_vis.html', 'w') as f:
        output = ' '.join(f"<span style='background-color: rgba(255, 0, 0, {weights_[i]})'>{int(nodes_[i])}</span>"
                          for i in range(len(nodes_)) if nodes_[i] != 0)
        f.write(output)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', default='', type=str,
                        help='Path to the graph file')
    parser.add_argument('--fmt', type=str, default='edgelist',
                        help="Format ('edgelist-Default', 'adjlist') of the input graph file. ")
    parser.add_argument('--output-dir', type=str, default="",
                        help='Path to save outputs, mainly the embedding file.')
    parser.add_argument('--dim', type=int, default=200,
                        help='Embedding dimension. Default is 200')
    parser.add_argument('--epochs', type=int, default=100,
                        help='The number of training epochs. Default is 100')
    parser.add_argument('--tr-rate', type=float, default=.15,
                        help='Use only tr-rate fraction of the edges for train-set, a value in (0, 1]. Default is 0.15')
    parser.add_argument('--dev-rate', type=float, default=0.2,
                        help='Use dev-rate fraction of the training set for dev-set, a value in [0, 1). Default is 0.2')
    parser.add_argument('--learning-rate', type=float, default=.0001,
                        help='The learning rate. Default is 0.0001')
    parser.add_argument('--dropout-rate', type=float, default=.5,
                        help='The dropout rate. Default is 0.5')
    parser.add_argument('--nbr-size', type=int, default=100,
                        help='The maximum neighborhood size. Default is 100')
    parser.add_argument('--directed', type=int, default=1,
                        help='Whether the graph is directed. 0 - undirected, 1- directed. Default is 1
                        ')
    parser.add_argument('--verbose', type=int,
                        default=1, help="Turn logging on or off - (0 - off, Any value - on). Default is 1")
    return parser.parse_args()