import torch
import time
import torch.nn.functional as F
from torch import nn as nn
from torch.nn.parameter import Parameter
from model.Dipole import Dip_c, Dip_g, Dip_l
from model.PKGAT import GATModel


class DiseasePredModel(nn.Module):
    def __init__(self, path_filtering, joint_impact, causal_analysis, input_dim, output_dim, hidden_dim, embed_dim, num_path, threshold, dropout, alpha_CAPF, device_id, bi_direction=False, device=torch.device("cuda")):
        super().__init__()

        self.path_filtering = path_filtering
        self.joint_impact = joint_impact
        self.causal_analysis = causal_analysis
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
        :param feature_index : patient multi-hot EHR medical records
        :param rel_index: extracted paths in the personalized knowledge graphs (PKGs)
        :param feat_index: patient feature index
        """

        lstm_out, h_time = self.dipole(feature_index)
        if only_dipole == True:
            return self.out_activation(lstm_out)
        else:
            lstm_out = self.Wlstm(lstm_out)
            if self.causal_analysis:
                g_i, g_c, g_t, path_attentions, causal_attentions = self.pkgat(rel_index, feat_index, h_time, self.path_filtering, self.joint_impact, self.causal_analysis)

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
            else:
                g_kg = self.pkgat(rel_index, feat_index, h_time, self.path_filtering, self.joint_impact, self.causal_analysis)
                final_lstm = p * lstm_out
                kg_out = self.Wkg(g_kg)
                final_kg = final_lstm + (1 - p) * kg_out
                final_kg = self.out_activation(final_kg)
                return final_kg
        
    def interpret(self, path_attentions, causal_attentions, top_k=1):
        """
        :param path_attentions: path attention weight in a sample
        :param causal_attentions: causal attention of each feature
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