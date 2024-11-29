import torch
import time
import torch.nn.functional as F
from torch import nn as nn
from torch.nn.parameter import Parameter
from model.Dipole import Dip_c, Dip_g, Dip_l
from model.PKGAT import GATModel


class DiseasePredModel(nn.Module):
    def __init__(self, model_type: str, input_dim, output_dim, hidden_dim, embed_dim, num_path, threshold, dropout, alpha_CAPF, device_id, bi_direction=False, device=torch.device("cuda")):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.bi_direction = bi_direction
        self.device = device

        self.Wlstm = nn.Linear(output_dim, output_dim, bias=False)
        self.dipole = Dip_g(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            bi_direction=self.bi_direction,
            device=self.device
        )

        self.Wkg = nn.Linear(output_dim, output_dim, bias=False)
        self.pkgat = GATModel(
            nfeat=input_dim,
            nemb=self.embed_dim,
            gat_layers=1,
            gat_hid=self.output_dim,
            dropout=dropout,
            alpha=alpha_CAPF,
            num_path=num_path,
            threshold=threshold,
            device_id=device_id
        )

        self.out_activation = nn.Sequential(
            nn.Linear(output_dim, output_dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, feature_index, rel_index, feat_index, only_dipole, p):
        """
        :param feature_index : tensor(bs, num_visit, num_feat) multi-hot vector x_t
        :param rel_index: tensor(bs, num_visit, num_feat, num_target, num_path, K, num_rel + 1)
        :param feat_index: (bs, num_visit, num_feat, num_feat)
        """

        lstm_out, h_time = self.dipole(feature_index)
        if only_dipole == True:
            return self.out_activation(lstm_out)
        else:
            lstm_out = self.Wlstm(lstm_out)
            g_i, g_c, g_t, path_attentions, causal_attentions = self.pkgat(feature_index, rel_index, feat_index, h_time)

            kg_out_i = self.Wkg(g_i)
            kg_out_c = self.Wkg(g_c)
            kg_out_t = self.Wkg(g_t)

            final_lstm = p * lstm_out
            final_i = final_lstm + (1 - p) * kg_out_i
            final_c = final_lstm + (1 - p) * kg_out_c
            final_t = final_lstm + (1 - p) * kg_out_t
            final_i = self.out_activation(final_i)
            final_c = self.out_activation(final_c)
            final_t = self.out_activation(final_t)

            return final_i, final_c, final_t, path_attentions, causal_attentions
        
    def interpret(self, path_attentions, causal_attentions, top_k=1):
        """
        :param path_attentions: (bs, num_visit, num_feat, num_target, num_path)
        :param causal_attentions: (bs, num_visit, max_feat)
        """
        bs, _, _ = causal_attentions.size()
        path_attentions = torch.einsum('bvf, bvftp -> bvftp', causal_attentions, path_attentions)
        path_attentions = path_attentions.view(bs, -1)

        sample_top_attn = []
        sample_top_index = []

        for batch_idx in range(bs):
            batch_path_attn = path_attentions[batch_idx, :]
            top_path = torch.topk(batch_path_attn, top_k)
            top_attn = top_path.values
            top_index = top_path.indices

            # print(f"Layer {layer_idx}, Batch {batch_idx}, Visit {visit_idx} - Top {top_k} Paths:")
            # for i, idx in enumerate(top_index):
            #     print(f"  Path {i}: Attention Value = {top_attn[idx]}")
            
            sample_top_attn.append(top_attn)
            sample_top_index.append(top_index)
        
        return sample_top_attn, sample_top_index