import torch as t
import torch.nn as nn


class RankingLoss:

    def __init__(self, model, margin=0.5):
        model_score = model.source_rep.mul(model.target_rep).sum(1)
        noise_score = model.source_rep.mul(model.negative_rep).sum(1)
        score = t.relu(margin - model_score + noise_score)
        self.loss = t.mean(score)


class Encoder(nn.Module):
    
    """
    The simplified Encoder of APN that only has an embedding component
    """

    def __init__(self, in_dim, out_dim, rate):
        super(Encoder, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rate = rate
        self._init()

    def _init(self):
        self.embedding = nn.Embedding(self.in_dim, self.out_dim, padding_idx=0)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.dropout = nn.Dropout(self.rate)

    def forward(self, x):
        return self.dropout(self.embedding(x).transpose(1, 2))


class GAP(nn.Module):
    
    """
    GAP model
    """

    def __init__(self, num_nodes, emb_dim, rate=0.5):
        super(GAP, self).__init__()
        self.num_nodes = num_nodes
        self.emb_dim = emb_dim
        self.rate = rate
        self._init()

    def _init(self):
        self.encoder = Encoder(in_dim=self.num_nodes + 1, out_dim=self.emb_dim, rate=self.rate)
        self.attention_parameter = nn.Parameter(t.FloatTensor(self.emb_dim, self.emb_dim))
        nn.init.xavier_normal_(self.attention_parameter)

    def _model_forward(self, source_nh, target_nh, source_mask, target_mask):
        # self.source_emb.shape = self.target_emb.shape = [batch_size, emb_dim, neighborhood_size]
        self.source_emb = self.encoder(source_nh)
        self.target_emb = self.encoder(target_nh)

        self.source_target_sim = self.source_emb.transpose(1, 2).matmul(
            self.attention_parameter.unsqueeze(0)).matmul(self.target_emb)
        self.source_target_sim = t.tanh(self.source_target_sim)

        source_attention_vec = t.mean(self.source_target_sim, dim=-1, keepdim=True)
        source_attention_vec = source_attention_vec + source_mask.unsqueeze(-1)
        self.source_attention_vec = t.softmax(source_attention_vec, dim=1)

        target_attention_vec = t.mean(self.source_target_sim, dim=1, keepdim=True).transpose(1, 2)
        target_attention_vec = target_attention_vec + target_mask.unsqueeze(-1)
        self.target_attention_vec = t.softmax(target_attention_vec, dim=1)

        self.source_rep = self.source_emb.matmul(self.source_attention_vec).squeeze()
        self.target_rep = self.target_emb.matmul(self.target_attention_vec).squeeze()

    def _noise_forward(self, negative_nh, negative_mask):
        self.neg_emb = self.encoder(negative_nh)

        self.source_neg_sim = self.source_emb.transpose(1, 2).matmul(
            self.attention_parameter.unsqueeze(0)).matmul(self.neg_emb)
        self.source_neg_sim = t.tanh(self.source_neg_sim)

        neg_attention_vec = t.mean(self.source_neg_sim, dim=1, keepdim=True).transpose(1, 2)
        neg_attention_vec = neg_attention_vec + negative_mask.unsqueeze(-1)
        self.neg_attention_vec = t.softmax(neg_attention_vec, dim=1)

        self.negative_rep = self.neg_emb.matmul(self.neg_attention_vec).squeeze()

    def forward(self, source_neighborhood=None, target_neighborhood=None, negative_neighborhood=None,
                source_mask=None, target_mask=None, negative_mask=None):
        self._model_forward(source_nh=source_neighborhood, target_nh=target_neighborhood, 
                            source_mask=source_mask, target_mask=target_mask)
        if negative_neighborhood is not None:
            self._noise_forward(negative_nh=negative_neighborhood, negative_mask=negative_mask)