import torch
import torch.nn.functional as F
from torch import nn as nn
import math


class Embedding(nn.Module):
    def __init__(self, num_feat, num_embed):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(num_feat, num_embed)

        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, feat_index):
        return self.embedding(feat_index)


class GraphCare(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_rel=11, num_feat=1992, num_visit=6, reduced_dim=256, gamma=0.1, dropout=0.3, max_feat=12, bidirectional=False, device=torch.device('cuda')):
        super(GraphCare, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_feat = num_feat
        self.num_visit = num_visit
        self.max_feat = max_feat

        self.emb = Embedding(self.input_dim, self.hidden_dim)

        self.W = nn.ParameterList()
        for _ in range(num_rel):
            self.W.append(nn.Parameter(torch.rand(self.hidden_dim)))

        self.W_v = nn.Parameter(torch.rand(size=(reduced_dim, self.hidden_dim)))
        self.W_r = nn.Parameter(torch.rand(size=(reduced_dim, self.hidden_dim)))
        self.b_v = nn.Parameter(torch.rand(reduced_dim))
        self.b_r = nn.Parameter(torch.rand(reduced_dim))
        self.W_alpha = nn.Parameter(torch.rand(self.num_feat))
        self.b_alpha = nn.Parameter(torch.rand(self.max_feat))
        self.W_beta = nn.Parameter(torch.rand(self.max_feat))
        self.b_beta = nn.Parameter(torch.rand(self.num_visit))

        self.lamb = [math.exp(-gamma * (self.num_visit - i)) for i in range(self.num_visit)]
        self.lamb = torch.tensor(self.lamb)
        self.lamb = self.lamb.cuda(1)

        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

        self.W_l = nn.Parameter(torch.rand(size=(reduced_dim, reduced_dim)))
        self.b_l = nn.Parameter(torch.rand(reduced_dim))
        self.relu = nn.ReLU(inplace=True)

        self.final_mlp = nn.Linear(2 * reduced_dim, self.output_dim)

        self.sigmod = nn.Sigmoid()

    def forward(self, feat_index, neighbor_index, rel_index):
        """
        :param feat_index: tensor(bs, num_visit, max_feat, num_feat) multi-hot vector
        :param neighbor_index: tensor(bs, num_visit, max_feat, max_target, num_feat)    one-hot vector
        :param rel_index: tensor(bs, num_visit, max_feat, max_target, num_rel)    multi-hot vector
        """
        #node-level attn
        alpha = torch.einsum('n, bvmn -> bvm', self.W_alpha, feat_index)
        alpha = self.softmax(alpha + self.b_alpha.view(1, 1, -1))
        
        #visit-level attn
        beta = self.tanh(torch.einsum('m, bvm -> bv', self.W_beta, alpha) + self.b_beta.view(1, -1))
        beta = torch.einsum('n, bn -> bn', self.lamb, beta)

        #node aggregation
        reduced_emb = torch.einsum('ih, rh -> ir', self.emb.embedding.weight, self.W_v) + self.b_v.view(1, -1)
        neighbor_emb = torch.einsum('bvmtn, nr -> bvmtr', neighbor_index, reduced_emb)
        attn = torch.einsum('bnm, bn -> bnm', alpha, beta)
        node_agg = torch.einsum('bnm, bnmtr -> bnmtr', attn, neighbor_emb)
        #edge aggregation
        edge = torch.stack([param for param in self.W], dim=0)
        reduced_edge = torch.einsum('nh, rh -> nr', edge, self.W_r) + self.b_r.view(1, -1)
        edge_agg = torch.einsum('bvftr, rd -> bvftd', rel_index, reduced_edge)

        node_agg = torch.sum(node_agg, dim=-2)
        edge_agg = torch.sum(edge_agg, dim=-2)
        agg = node_agg + edge_agg
        h = self.relu(torch.einsum('dd, bvfd -> bvfd', self.W_l, agg) + self.b_l.view(1, -1))
        
        h_g = torch.sum(h, dim=-2)
        h_g = torch.sum(h_g, dim=-2)

        h_p = torch.sum(torch.einsum('bvmn, bvmr -> bvnr', feat_index, h), dim=-2)
        h_p = torch.sum(h_p, dim=-2)

        z_joint = torch.cat([h_g, h_p], dim=-1)
        output = self.sigmod(self.final_mlp(z_joint))

        return output