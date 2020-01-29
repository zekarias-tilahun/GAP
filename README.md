# GAP
Implementation of GAP: Graph Neighborhood Attentive Pooling, https://arxiv.org/abs/2001.10394. A context-sensitve graph (network) representation learning algorithm that relies only on the structure of the graph.

### Requirements!
  - Python 3.6+
  - PyTorch 1.3.1+
  - Numpy 1.17.2+
  - Networkx 2.3+
## Usage
#### Example usage
```sh
$ cd GAP
$ python src/gap.py
```
OR
```sh
$ cd GAP
$ bash run.sh
```

### Input Parameters


`--input:`
A path to a graph file. Default is ```../data/cora/graph.txt```

`--fmt:`
The format of the input graph, either ```edgelist``` or ```adjlist```. Default is ```edgelist```

`--output-dir:`
A path to a directory to save a trained model. Default is ```../data/cora/outputs```

`--dim:`
The size (dimension) of nodes embedding (representation) vector. Default is 200.

`--epochs:`
The number of epochs. Default is 100.

`--tr-rate:`
Training rate, i.e. the fraction of edges to be used as a training set. A value in (0, 1]. Default is .15. The remaining fraction of edges (```1 - tr-rate```) will be saved in the directory specified by ```--ouput-dir``` argument.

`--dev-rate:`
Development rate, i.e. the fraction of the training edges to be used as a development (validation) set. A value in (0, 1]. Default is 0.2.

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

| Algorithm     | Training ratio||||
|---------------|----------|----------|----------|----------|
|               | **15%**  | **35%**  | **55%**  | **75%**  |
| DeepWalk      | 56.0     | 70.2     | 80.1     | 85.3     |
| Node2Vec      | 55.0     | 66.4     | 77.6     | 85.6     |
| AttentiveWalk | 64.2     | 81.0     | 87.1     | 92.4     |
| TriDnr        | 85.9     | 90.5     | 91.3     | 93.0     |
| CENE          | 72.1     | 84.6     | 89.4     | 93.9     |
| CANE          | 86.8     | 92.2     | 94.6     | 95.6     |
| DMTE          | 91.3     | 93.7     | 96.0     | 97.4     |
| SPLITTER      | 65.4     | 73.7     | 80.1     | 83.9     |
| GAP           | **95.8** | **97.1** | **97.6** | **97.8** |


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