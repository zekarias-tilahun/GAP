from sklearn import metrics, cluster

import networkx as nx
import pandas as pd
import numpy as np
import argparse

import gap_helper


def nmi_ami(com_path, emb_path, seed=0):
    gap_helper.log(f'Reading ground truth communities from {com_path}')
    com_df = pd.read_csv(com_path, header=None, sep=r'\s+', names=['node', 'com'], index_col=0)
    gap_helper.log(f'Reading embeddings from {emb_path}')
    emb_df = pd.read_csv(emb_path, header=None, sep=r'\s+', index_col=0)

    gap_helper.log('Building features')
    labeled_features = com_df.merge(emb_df, left_index=True, right_index=True)
    ground_truth = labeled_features.com.values
    features = labeled_features.values[:, 1:]
    num_com = len(set(ground_truth))

    gap_helper.log(f'Learning to identify {num_com} clusters using spectral clustering')
    clustering = cluster.SpectralClustering(
        n_clusters=num_com, assign_labels="discretize", random_state=seed)
    predictions = clustering.fit(features).labels_
    nmi = metrics.normalized_mutual_info_score(ground_truth, predictions, average_method='arithmetic')
    ami = metrics.adjusted_mutual_info_score(ground_truth, predictions, average_method='arithmetic')
    return nmi, ami


def auc_score(is_dev=True, u_embed=None, v_embed=None, test_edges=None):
    # Adopted from CANE: https://github.com/thunlp/CANE/blob/master/code/auc.py
    if is_dev:
        nodes = list(range(u_embed.shape[0]))
        test_edges = list(zip(range(u_embed.shape[0]), range(v_embed.shape[0])))
    else:
        nodes = list({n for edge in test_edges for n in edge})

    def get_random_index(u, v, lookup=None):
        while True:
            node = np.random.choice(nodes)
            if node != u and node != v:
                if lookup is None:
                    return node
                elif node in lookup:
                    return node

    hit = 0.
    counter = 0.
    for i in range(len(test_edges)):
        if is_dev:
            u = v = i
            j = get_random_index(u=i, v=i)
        else:
            u = test_edges[i][0]
            v = test_edges[i][1]
            if u not in u_embed or v not in u_embed:
                continue
            j = get_random_index(u=u, v=v, lookup=v_embed)

        u_emb = u_embed[u]
        v_emb = v_embed[v]
        j_emb = v_embed[j]

        pos_score = np.dot(u_emb, v_emb)
        neg_score = np.dot(u_emb, j_emb)

        if pos_score > neg_score:
            hit += 1.
        elif pos_score == neg_score:
            hit += 0.5
        counter += 1.

    return hit / counter


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb-path', required=True, type=str, help='Path to the embedding file')
    parser.add_argument('--te-path', type=str, default='', help='Path to the test edges file')
    parser.add_argument('--com-path', type=str, default='', help='Path to the ground truth community file')
    parser.add_argument('--verbose', type=bool, default=True)
    return parser.parse_args()


def main(args):
    
    gap_helper.VERBOSE = args.verbose
    if args.te_path == '' and args.com_path == '':
        gap_helper.log('Atleast a path to test edges file or ground truth community file should be specified',
                       level=gap_helper.ERROR)
    else:
        results = {}
        if args.te_path != '':
            gap_helper.log('Running link prediction', level=gap_helper.INFO)
            embeddings = gap_helper.read_embedding(args.emb_path)
            test_graph = nx.read_edgelist(args.te_path, nodetype=int)
            scores = []
            gap_helper.log('Computing AUC')
            for i in range(10):
                score = auc_score(is_dev=False, u_embed=embeddings, v_embed=embeddings,
                                  test_edges=list(test_graph.edges()))
                scores.append(score)
            avg = np.mean(scores)
            std = np.std(scores)
            gap_helper.log(f"Average auc score = {avg}")
            gap_helper.log(f"Standard deviation = {std}")
            results['link_prediction'] = avg, std
        if args.com_path != '': 
            gap_helper.log('Running node clustering')
            nmi, ami = nmi_ami(com_path=args.com_path, emb_path=args.emb_path)
            gap_helper.log(f'NMI: {nmi}')
            gap_helper.log(f'AMI: {ami}')
            results['node_clustering'] = nmi, ami
        return results

if __name__ == '__main__':
    main(parse())