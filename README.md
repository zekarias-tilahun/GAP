# GAP
Implementation of GAP: Graph Neighborhood Attentive Pooling, https://arxiv.org/abs/2001.10394. A context-sensitve graph (network) representation learning algorithm that relies only on the structure of the graph.

### Requirements!
  - Python 3.6+
  - PyTorch 1.3.1+
  - Numpy 1.17.2+
  - Networkx 2.3+
  
#### Optional Requirment 
for evaluating node-clustering performance of algorithms
  - scikit-learn 0.21.3+

## Usage
#### Example usage
```sh
$ cd src
$ python gap.py
```
OR
```sh
$ bash run.sh
```

### Input Arguments


`--input:`
A path to a graph file. Default is ```../data/cora/graph.txt```

`--fmt:`
The format of the input graph, either ```edgelist``` or ```adjlist```. Default is ```edgelist```

`--output-dir:`
A path to a directory to save intermediate and final outputs of GAP. Default is ```../data/cora/outputs```

`--dim:`
The size (dimension) of nodes' embedding (representation) vector. Default is 200.

`--epochs:`
The number of epochs. Default is 100.

`--tr-rate:`
Training rate, i.e. the fraction of edges to be used as a training set. A value in (0, 1]. Default is .15. The remaining fraction of edges (```1 - tr_rate```), test edges, will be saved in the directory specified by ```--ouput-dir``` argument.

`--dev-rate:`
Development rate, i.e. the fraction of the training set to be used as a development (validation) set. A value in [0, 1). Default is 0.2.

`--learning-rate:`
Learning rate, a value in [0, 1]. Default is 0.0001

`--dropout-rate:`
Dropout rate, a value in [0, 1]. Deafult is 0.5

`--nbr-size:`
The number of neighbors to be sampled. Default is 100.

`--directed:`
Whether the graph is directed or not. 1 for directed and 0 for undirected. Default is 1.

`--verbose:`. 
Whether to turn on a verbose logger or not. 1 is on and 0 is off. Default is 1.

Some Results
------------
An excerpt of the reported results for link-prediction on the cora dataset.
<table>
  <tr>
    <th rowspan="2">Algorithm</th>
    <th colspan="4">Training ratio</th>
  </tr>
  <tr>
    <td><b>0.15</b></td>
    <td><b>0.35</b></td>
    <td><b>0.55</b></td>
    <td><b>0.75</b></td>
  </tr>
  <tr>
    <td><a href="https://arxiv.org/abs/1403.6652">DeepWalk</a></td>
    <td>56.0</td>
    <td>70.2</td>
    <td>80.1</td>
    <td>85.3</td>
  </tr>
  <tr>
    <td><a href="https://arxiv.org/abs/1607.00653">Node2Vec</td>
    <td>55.0</td>
    <td>66.4</td>
    <td>77.6</td>
    <td>85.6</td>
  </tr>
  <tr>
    <td><a href="https://arxiv.org/abs/1710.09599">AttentiveWalk</td>
    <td>64.2</td>
    <td>81.0</td>
    <td>87.1</td>
    <td>92.4</td>
  </tr>
  <tr>
    <td><a href="https://dl.acm.org/doi/10.5555/3060832.3060886">TriDnr</a></td>
    <td>85.9</td>
    <td>90.5</td>
    <td>91.3</td>
    <td>93.0</td>
  </tr>
  <tr>
    <td><a href="https://arxiv.org/abs/1610.02906">CENE</a></td>
    <td>72.1</td>
    <td>84.6</td>
    <td>89.4</td>
    <td>93.9</td>
  </tr>
  <tr>
    <td><a href="https://www.aclweb.org/anthology/P17-1158/">CANE</a></td>
    <td>86.8</td>
    <td>92.2</td>
    <td>94.6</td>
    <td>95.6</td>
  </tr>
  <tr>
    <td><a href="https://dl.acm.org/doi/10.5555/3327757.3327858">DMTE</a></td>
    <td>91.3</td>
    <td>93.7</td>
    <td>96.0</td>
    <td>97.4</td>
  </tr>
  <tr>
    <td><a href="https://arxiv.org/abs/1905.02138">SPLITTER</a></td>
    <td>65.4</td>
    <td>73.7</td>
    <td>80.1</td>
    <td>83.9</td>
  </tr>
  <tr>
    <td><a href=".">GAP</a></td>
      <td><b>95.8</b></td>
      <td><b>97.1</b></td>
      <td><b>97.6</b></td>
      <td><b>97.8</b></td>
  </tr>
</table>


Citing
------
```
@misc{kefato2020graph,
    title={Graph Neighborhood Attentive Pooling},
    author={Zekarias T. Kefato and Sarunas Girdzijauskas},
    year={2020},
    eprint={2001.10394},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```