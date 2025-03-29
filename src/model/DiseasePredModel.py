import torch
import torch.nn as nn
from model.lstm import LSTM_Model
from model.PKGAT_2 import GATModel
import pickle


class DiseasePredModel(nn.Module):
    """
    A hybrid model for disease prediction combining time-series EHR analysis and knowledge graph reasoning.
    """
    def __init__(self, path_filtering, joint_impact, causal_analysis, input_dim, output_dim, 
                 hidden_dim, embed_dim, threshold, dropout, alpha_CAPF, device_id, 
                 bi_direction=False, device=torch.device("cuda")):
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

        # Initialize time-series module
        self.Wlstm = nn.Linear(output_dim, output_dim, bias=False)
        self.dipole = LSTM_Model(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            bi_direction=self.bi_direction
        )

        # Initialize personalized knowledge graph module (PFCA)
        self.Wkg = nn.Linear(output_dim, output_dim, bias=False)
        self.pkgat = GATModel(
            nfeat=input_dim,
            nemb=self.embed_dim,
            gat_layers=1,
            gat_hid=self.output_dim,
            dropout=dropout,
            alpha=alpha_CAPF,
            threshold=threshold,
            device_id=device_id
        )

        self.out_activation = nn.Sequential(
            nn.Linear(output_dim + embed_dim, output_dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, feature_index, feat_index, path_index, path_node, path_relation, only_dipole, p):
        """
        :param feature_index : patient multi-hot EHR medical records
        :param rel_index: extracted paths in the personalized knowledge graphs (PKGs)
        :param feat_index: patient feature index
        """
        # Get patient time-series embeddings
        lstm_out, h_time = self.dipole(feature_index)
        
        if only_dipole:
            return self.out_activation(lstm_out)
        
        # Combine time-series and graph information
        lstm_out = self.Wlstm(lstm_out)
        if self.causal_analysis:
            g_i, g_c, g_t, path_attentions, causal_attentions = self.pkgat(
                feat_index, path_index, path_node, path_relation, h_time, self.path_filtering, 
                self.joint_impact, self.causal_analysis
            )
            final_i = self.out_activation(torch.cat([lstm_out, g_i], dim = -1))
            final_c = self.out_activation(torch.cat([lstm_out, g_c], dim = -1))
            final_t = self.out_activation(torch.cat([lstm_out, g_t], dim = -1))

            return final_i, final_c, final_t, path_attentions, causal_attentions
        else:
            # Standard prediction without causal analysis
            g_kg = self.pkgat(feat_index, path_index, path_node, path_relation, h_time, self.path_filtering, 
                             self.joint_impact, self.causal_analysis)
            final_kg = self.out_activation(torch.cat([lstm_out, g_kg], dim = -1))
            
            return final_kg
