from collections import namedtuple
import networkx as nx
import numpy as np
import random
import os

import gap_helper

import torch

random.seed(40)
np.random.seed(40)


def source_targets(edges): return [list(lor_edges) for lor_edges in zip(*edges)]


def _get_mask(arr):
    """
    Creates a mask for the input array. It will be used to mask the effect of zero-padding in the softmax
    computation of the attention weights.
    :param arr: The zero padded neighborhood array
    :return: A mask array
    """
    neg_inf = -99999999.
    mask = (arr != 0)
    mask = mask.astype('float')
    mask[mask == 0] = neg_inf
    mask[mask != neg_inf] = 0
    return mask


def _sample_neighborhood(nbr_size, g):
    """
    Samples a fixed number of (nbr_size) neighbors for every node in the graph.
    For nodes with smaller number of neighbors, the neighborhood will be padded with zero

    :param nbr_size: The neighborhood size
    :param g: The input graph
    :return: An n x nbr_size neighborhood matrix
    """
    
    nodes = list(g.nodes())
    neighborhood_matrix = np.zeros(shape=(len(nodes), nbr_size))
    for n in nodes:
        nbrs = list(set(nx.all_neighbors(g, n)) | {n})
        arr = np.array(nbrs) + 1
        if arr.shape[0] > nbr_size:
            arr = np.random.choice(arr, size=nbr_size, replace=False)
        neighborhood_matrix[n, :arr.shape[0]] = arr
    return neighborhood_matrix


def relable_nodes(graph):
    gap_helper.log('Node relabeling ...')
    nodes = sorted(graph.nodes())
    node_ids = range(len(nodes))
    node_id_map = dict(zip(nodes, node_ids))
    id_node_map = dict(zip(node_ids, nodes))
    return nx.relabel_nodes(graph, node_id_map), id_node_map


class Data:

    def __init__(self, args):
        self._args = args
        self._hold_out = args.tr_rate < 1.
        self._use_dev = args.dev_rate > 0
        self._batch_size = 64
        self._read_graph()
        self._train_test_split()
        self._build_neighborhood()
        self._negative_sample()
        self._build_train_dev_batches()

    def _read_graph(self):
        args = self._args
        self._reader = nx.read_adjlist if args.fmt == 'adjlist' else nx.read_edgelist
        self._creator = nx.DiGraph if args.directed else nx.Graph
        gap_helper.log(f'Reading graph from {args.input}')
        self.graph = self._reader(path=args.input, create_using=self._creator, nodetype=int)
        self.graph, self.id_to_node = relable_nodes(self.graph)
        self.num_nodes = self.graph.number_of_nodes()
        gap_helper.log(f'Number of nodes {self.num_nodes}')
        gap_helper.log(f'Number of edges {self.graph.number_of_edges()}')
        
    def _train_test_split(self):
        def split(train_rate):
            edges = list(self.graph.edges())
            train_size = int(len(edges) * train_rate)
            random.shuffle(edges)
            train_edges = edges[:train_size]
            test_edges = edges[train_size:]
            return source_targets(train_edges), source_targets(test_edges)
        
        args = self._args
        self._test_nodes = []
        if self._hold_out:
            splits = split(args.tr_rate)
            self._train_sources, self._train_targets = splits[0]
            if args.output_dir != '':
                test_sources, test_targets = splits[1]
                self._test_nodes = set(test_sources) | set(test_targets)
                path = os.path.join(args.output_dir, f'test_graph_{int(args.tr_rate * 100)}.txt')
                gap_helper.log(f"Persisting test data to {path} and the number of test points is  {len(test_sources)}")
                nx.write_edgelist(self._creator(list(zip(test_sources, test_targets))), path=path, data=False)
        else:
            gap_helper.log('No test data is persisted')
            self._train_sources, self._train_targets = source_targets(self.graph.edges())

        self._train_nodes = set(self._train_sources) | set(self._train_targets)

    def _build_neighborhood(self):
        args = self._args
        self._neighborhood_matrix = _sample_neighborhood(nbr_size=args.nbr_size, g=self.graph)
        
        """
        Maskings
            Masking 1. The first masking ensures that no node from the test set is sampled in the neighbhorhood of any node.
            Masking 2. The second one is used for zero-padding, used during the softmax computation
        """
        
        # Masking 1
        self.mask_nodes = self._test_nodes if self._hold_out else None
        if self.mask_nodes is not None:
            msk = np.in1d(self._neighborhood_matrix, self.mask_nodes)
            self._neighborhood_matrix[msk.reshape(self._neighborhood_matrix.shape)] = 0

        # Masking 2
        self._mask_matrix = torch.FloatTensor(_get_mask(self._neighborhood_matrix))
        self._neighborhood_matrix = torch.LongTensor(self._neighborhood_matrix)
        
    def _negative_sample(self):
        def get_negative_node_to(u, v):
            while True:
                node = self.node_dist_table[random.randint(0, len(self.node_dist_table) - 1)]
                if node != u and node != v:
                    return node

        gap_helper.log('Sampling negative nodes')
        degree = {node: int(1 + self.graph.degree(node) ** 0.75) for node in self.graph.nodes()}
        # node_dist_table is equivalent of the uni-gram distribution table in the word2vec implementation
        self.node_dist_table = [node for node, new_degree in degree.items() for _ in range(new_degree)]

        sources, targets = self._train_sources, self._train_targets
        src, trg, neg = [], [], []

        for i in range(len(sources)):
            neg_node = get_negative_node_to(sources[i], targets[i])
            src.append(sources[i])
            trg.append(targets[i])
            neg.append(neg_node)

        self._train_sources, self._train_targets, self._train_negatives = np.array(src), np.array(trg), np.array(neg)
        
    def batch_iterator(self):
        for i in range(0, size, batch_size):
            btc = build_batch(start=i)
            batches.append(namedtuple('Batch', btc.keys())(*btc.values()))
            
    def _create_train_dev_indices(self):
        args = self._args
        self._dev_indices = []
        self._train_indices = np.arange(self._train_sources.shape[0])
        if self._use_dev:
            dev_size = int(len(self._train_sources) * args.dev_rate)
            gap_helper.log(f'Number of dev points: {dev_size}')
            self._dev_indices = np.arange(dev_size)
            self._train_indices = np.arange(dev_size, self._train_sources.shape[0])
            
    def _fetch_current_batch(self, start, size, sources, targets, negatives):
        """
        Prepares model inputs using specified indices and organizes them into a batch

        :param start: Starting index 
        :param size: The number of edges
        :param sources: Source nodes 
        :param targets: Target nodes
        :param negatives: Negative nodes
        :param batch_size: Batch size
        :return: A Batch input
        """
        end = start + self._batch_size if size - start > self._batch_size else size
        src, trg, neg = sources[start:end], targets[start:end], negatives[start:end]
        batch = {'source': torch.LongTensor(src), 'target': torch.LongTensor(trg), 'negative': neg, 
                 'source_neighborhood': self._neighborhood_matrix[src], 'target_neighborhood': self._neighborhood_matrix[trg],
                 'negative_neighborhood': self._neighborhood_matrix[neg], 'source_mask': self._mask_matrix[src],
                 'target_mask': self._mask_matrix[trg],
                 'negative_mask': self._mask_matrix[neg]}
        return namedtuple('Batch', batch.keys())(*batch.values())
    
    def _build_batch_iterator(self, idx):
        gap_helper.log('Building batch iterator')
        sources, targets, negatives = self._train_sources[idx], self._train_targets[idx], self._train_negatives[idx]
        size = idx.shape[0]
        for i in range(0, size, self._batch_size):
            yield self._fetch_current_batch(start=i, size=size, sources=sources, targets=targets, negatives=negatives)

    def _build_batches(self, idx):
        gap_helper.log('Building in memory batches')
        batches = []
        sources, targets, negatives = self._train_sources[idx], self._train_targets[idx], self._train_negatives[idx]
        size = idx.shape[0]
        for i in range(0, size, self._batch_size):
            batch = self._fetch_current_batch(start=i, size=size, sources=sources, targets=targets, negatives=negatives)
            batches.append(batch)
        return batches
        
    def _build_train_dev_batches(self):
        self.dev_inputs = []
        self._create_train_dev_indices()
        builder = self._build_batches if self.graph.number_of_edges() < 100000 else self._build_batch_iterator
        if self._use_dev:
            self.dev_inputs = builder(idx=self._dev_indices)
        self.train_inputs = builder(idx=self._train_indices)