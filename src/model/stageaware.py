import torch
import torch.nn.functional as F
from torch import nn as nn
from model.Dipole import Dip_g


class Embedding(nn.Module):
    def __init__(self, num_feat, num_embed):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(num_feat, num_embed)

        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, feat_index):
        return self.embedding(feat_index)


class StageAware(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_rel=11, num_feat=1992, num_visit=6, max_feat = 12, max_target=8, Lambda=0.1, bidirectional=True, device=torch.device('cuda')):
        super(StageAware, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_feat = num_feat
        self.num_visit = num_visit
        self.num_rel = num_rel
        self.max_feat = max_feat
        self.max_target = max_target

        self.emb = Embedding(self.input_dim, self.hidden_dim)

        self.r = nn.ParameterList()
            self.r.append(nn.Parameter(torch.rand(self.hidden_dim)))

        self.W = nn.ParameterList()
        for _ in range(num_rel):
            self.W.append(nn.Parameter(torch.rand(size=(self.hidden_dim, self.hidden_dim))))
            nn.init.xavier_uniform_(self.W[-1].data, gain=1.414)

        self.rnn = nn.GRU(input_dim, hidden_dim, 1, bidirectional = bidirectional)
        self.rnn_final = nn.GRU(hidden_dim, output_dim, 1, bidirectional = bidirectional)
        self.final = nn.Sequential(
            nn.Linear(2 * output_dim, output_dim),
            nn.Sigmoid()
        )

        self.w_r = nn.Parameter(torch.rand(2 * hidden_dim))
        self.W_r = nn.Parameter(torch.rand(size=(hidden_dim, hidden_dim)))
        self.softmax = nn.Softmax(dim=-1)
        self.leakyrelu = nn.LeakyReLU()

        self.W_n = nn.Parameter(torch.rand(size=(hidden_dim, hidden_dim)))
        self.w_n = nn.Parameter(torch.rand(3 * hidden_dim))

        self.Lambda = Lambda

    def forward(self, features, feat_index, neighbor_index, rel_index, h_time):
        """
        :param features: patient multi-hot EHR medical records
        :param feat_index: patient feature index
        :param neighbor_index: prediction targets multi-hot vector
        :param rel_index: relations types between patient features and target features
        """
        h_time = torch.mean(h_time, dim=1)
        hidden_state_1 = h_time.unsqueeze(dim = 1).repeat(1, self.num_rel, 1)

        ## relation-level attn
        r = torch.stack([param for param in self.r], dim = 0)
        bs = features.size(0)
        r = r.unsqueeze(dim = 0).repeat(bs, 1, 1)
        a_r_i = torch.cat([torch.einsum('dd, brd -> brd', self.W_r, r), torch.einsum('dd, brd -> brd', self.W_r, hidden_state_1)], dim = -1)
        a_r_i = torch.einsum('d, brd -> br', self.w_r, a_r_i)
        a_r_i = self.softmax(self.leakyrelu(a_r_i))
        a_r_i = a_r_i.unsqueeze(dim=1).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, self.num_visit, self.max_feat, self.max_target, 1)
        a_r_i = torch.einsum('bvftr, bvftr -> bvftr', rel_index, a_r_i)

        ##node-level attn
        feat_emb = torch.einsum('bvmf, fd -> bvmd', feat_index, self.emb.embedding.weight)
        neighbor_emb = torch.einsum('bvmtf, fd -> bvmtd', neighbor_index, self.emb.embedding.weight)
        a_n_3 = torch.einsum('dd, bd -> bd', self.W_n, h_time).unsqueeze(dim=1).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, self.num_visit, self.max_feat, self.max_target, 1)
        a_n = torch.cat([torch.einsum('dd, bvfd -> bvfd', self.W_n, feat_emb).unsqueeze(dim=-2).repeat(1, 1, 1, self.max_target, 1), torch.einsum('dd, bvftd -> bvftd', self.W_n, neighbor_emb), a_n_3], dim=-1)
        a_n = torch.einsum('d, bvftd -> bvft', self.w_n, a_n)
        a_n = self.softmax(self.leakyrelu(a_n))

        #message passing
        weights = [param for param in self.W]
        weights = torch.stack(weights, dim=0)
        weights = weights.view(self.num_rel, -1)
        rel_index = rel_index.view(-1, self.num_rel)
        rel_weight = torch.mm(rel_index, weights)
        rel_weight = rel_weight.view(bs, self.num_visit, self.max_feat, self.max_target, self.hidden_dim, self.hidden_dim)
        msg = torch.einsum('bvftdd, bvftd -> bvftd', rel_weight, neighbor_emb)

        #hierarchical message aggregation
        h_n = torch.einsum('bvft, bvftd -> bvftd', a_n, msg)
        h_n = h_n.unsqueeze(dim=-2).repeat(1, 1, 1, 1, self.num_rel, 1)
        h_n = torch.einsum('bvftr, bvftrd -> bvftd', a_r_i, h_n)
        h_n = torch.sum(h_n, dim=-2)
        h_n = self.Lambda * feat_emb + (1 - self.Lambda) * h_n
        h_g = torch.sum(h_n, dim=-2)
        
        output, _ = self.rnn_final(h_g.permute(1, 0, 2))
        output = output.permute(1, 0, 2)
        output = torch.mean(output, dim = 1)

        return self.final(output)