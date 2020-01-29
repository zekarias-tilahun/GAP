cd src
python gap.py --input ../data/cora/graph.txt --output-dir ../data/cora/outputs/ --tr-rate .15 --epochs 100
python gap_evaluate.py --emb-path ../data/cora/outputs/gap_global_15.emb --te-path ../data/cora/outputs/test_graph_15.txt
