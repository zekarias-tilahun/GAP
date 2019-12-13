from collections import namedtuple
from itertools import tee
from gap_data import Data

import torch

import numpy as np

import gap_evaluate
import gap_helper
import gap_model

import os


def to_cpu_tensor(tensor, device):
    if device == 'cpu':
        return tensor
    return tensor.cpu().data.numpy()


class GapWrapper:

    def __init__(self, args):
        self._args = args
        self.data = Data(args)
        self.loss_fun = gap_model.RankingLoss
        self.model = None
        self.context_embedding = {}
        self.global_embedding = {}
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        gap_helper.log(f'Running GAP on a {self.device} machine')

    def _validate(self, dev_batches):
        args = self._args
        losses = []
        aucs = []
        for batch in dev_batches:
            source_embed, target_embed = self._infer(batch=batch)
            val_crt = self.loss_fun(self.model)
            val_auc = gap_evaluate.auc_score(u_embed=source_embed, v_embed=target_embed)
            losses.append(to_cpu_tensor(val_crt.loss, self.device))
            aucs.append(val_auc)
        return np.mean(losses), np.mean(aucs)

    def train(self):
        args = self._args
        self.model = gap_model.GAP(num_nodes=self.data.num_nodes, emb_dim=args.dim)
        self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)
        if isinstance(self.data.train_inputs, list):
            """
            In Memory batches
            """
            train_inputs = self.data.train_inputs
            dev_inputs = self.data.dev_inputs
        else:
            """
            We create multiple copies of the training and dev batche iterators.
            Useful when the training input is large, > 100000 edges
            """
            train_inputs = tee(self.data.train_inputs, args.epochs)
            dev_inputs = tee(self.data.dev_inputs, args.epochs)
            
        for epoch in range(args.epochs):
            train_batches = train_inputs if isinstance(train_inputs, list) else train_inputs[epoch]
            for batch in train_batches:
                self._infer(batch)
                criterion = self.loss_fun(self.model)
                optimizer.zero_grad()
                criterion.loss.backward()
                optimizer.step()

            if args.dev_rate > 0:
                val_loss, val_auc = self._validate(dev_inputs if isinstance(dev_inputs, list) else dev_inputs[epoch])
                gap_helper.log('Epoch: {}/{} training loss: {:.5f} validation loss: {:.5f} validation AUC: {:.5f}'.format(
                    epoch + 1, args.epochs, criterion.loss.data, val_loss, val_auc))
            else:
                gap_helper.log("Epoch {}/{} training loss = {:.5f}".format(epoch + 1, args.epochs, criterion.loss.data))

    def _infer(self, batch, use_negative=True):
        args = self._args
        device = self.device
        if use_negative:
            neg_nh, neg_msk = batch.negative_neighborhood.to(device), batch.negative_mask.to(device)
        else:
            neg_nh, neg_msk = None, None

        self.model(source_neighborhood=batch.source_neighborhood.to(device),
                   target_neighborhood=batch.target_neighborhood.to(device),
                   negative_neighborhood=neg_nh, source_mask=batch.source_mask.to(device),
                   target_mask=batch.target_mask.to(device),
                   negative_mask=neg_msk)
        return to_cpu_tensor(self.model.source_rep, device), to_cpu_tensor(self.model.target_rep, device)

    def infer_embeddings(self, agg=lambda l: np.mean(l, axis=0)):
        def populate_context_embedding(source_nodes, target_nodes, source_emb, target_emb):
            def apply(u_emb, u):
                u_name = self.data.id_to_node[u]
                if u in self.context_embedding:
                    self.context_embedding[u_name].append(u_emb)
                else:
                    self.context_embedding[u_name] = [u_emb]
            for i in range(len(source_nodes)):
                apply(source_emb[i], int(to_cpu_tensor(source_nodes[i], self.device)))
                apply(target_emb[i], int(to_cpu_tensor(target_nodes[i], self.device)))
        
        for batch in self.data.train_inputs + self.data.dev_inputs:
            src_rep, trg_rep = self._infer(batch=batch, use_negative=False)
            populate_context_embedding(batch.source, batch.target, src_rep, trg_rep)

        for node in self.context_embedding:
            emb = np.array(self.context_embedding[node])
            self.global_embedding[node] = emb if len(emb.shape) == 1 else agg(emb)

    def save_embeddings(self):
        args = self._args
        if args.output_dir != '':
            suffix = '' if args.tr_rate == 1 else f'_{str(int(args.tr_rate * 100))}'
            path = os.path.join(args.output_dir, f'gap_context{suffix}.emb')
            gap_helper.log(f'Saving context embedding to {path}')
            with open(path, 'w') as f:
                for node in self.context_embedding:
                    for emb in self.context_embedding[node]:
                        output = '{} {}\n'.format(node, ' '.join(str(val) for val in emb))
                        f.write(output)

            path = os.path.join(args.output_dir, f'gap_global{suffix}.emb')
            gap_helper.log(f'Saving aggregated global embedding to {path}')
            with open(path, 'w') as f:
                for node in self.global_embedding:
                    output = '{} {}\n'.format(node, ' '.join(str(val) for val in self.global_embedding[node]))
                    f.write(output)

def main(args):
    gap_helper.VERBOSE = False if args.verbose == 0 else True
    wrapper = GapWrapper(args)
    wrapper.train()
    if args.output_dir != '':
        wrapper.infer_embeddings()
        wrapper.save_embeddings()


if __name__ == '__main__':
    main(gap_helper.parse_args())