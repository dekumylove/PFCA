import torch
import time
import torch.nn.functional as F
from torch import nn as nn
from torch.nn.parameter import Parameter
from model.lstm import LSTM_Model
from model.Dipole import Dip_c, Dip_g, Dip_l
from model.retain import Retain
from model.graphcare import GraphCare
from model.medpath import MedPath
from model.PKGAT import GATModel
from model.stageaware import StageAware
from model.HAP import HAP


class DiseasePredModel(nn.Module):
    def __init__(self, model_type: str, input_dim, output_dim, hidden_dim, embed_dim, num_path, threshold, dropout, alpha_CAPF, gamma_GraphCare, lambda_HAR, device_id, bi_direction=False, device=torch.device("cuda")):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.bi_direction = bi_direction
        self.device = device

        self.Wlstm = nn.Linear(output_dim, output_dim, bias=False)
        if model_type == "Dip_l":
            self.dipole = Dip_l(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                bi_direction=self.bi_direction,
                device=self.device
            )
        elif model_type == "Dip_c":
            self.dipole = Dip_c(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                max_timesteps=10,
                bi_direction=self.bi_direction,
                device=self.device
            )
        elif model_type == "Dip_g":
            self.dipole = Dip_g(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                bi_direction=self.bi_direction,
                device=self.device
            )
        elif model_type == "Retain":
            self.dipole = Retain(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                device=self.device
            )
        elif model_type == "LSTM": # LSTM
            self.dipole = LSTM_Model(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                bi_direction=self.bi_direction,
                activation="sigmoid",
            )
        elif model_type == 'HAP':
            self.dipole = HAP(
                embed_dim = embed_dim,
                hidden_dim = hidden_dim,
                output_dim = output_dim
            )
        else:
            self.dipole = Dip_g(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                bi_direction=self.bi_direction,
                device=self.device
            )

        self.Wkg = nn.Linear(output_dim, output_dim, bias=False)

        if model_type == 'GraphCare':
            self.pkgat = GraphCare(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                dropout=dropout,
                gamma=gamma_GraphCare
            )
        elif model_type == 'MedPath':
            self.pkgat = MedPath(
                nfeat=self.input_dim,
                nemb=self.hidden_dim,
                gat_hid=self.output_dim,
                gat_layers=1
            )
        elif model_type == "StageAware":
            self.pkgat = StageAware(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                Lambda=lambda_HAR,
                device=self.device
            )
        else:
            self.pkgat = GATModel(
                nfeat=input_dim,
                nemb=self.embed_dim,
                gat_layers=1,
                gat_hid=self.output_dim,
                dropout=0.1,
                alpha=alpha_CAPF,
                num_path=num_path,
                threshold=threshold,
                device_id=device_id
            )

        self.out_activation = nn.Sequential(
            nn.Linear(output_dim, output_dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, p_, p2c, feature_index, rel_index, neighbor_index, feat_index, only_dipole, p):
        
        lstm_out, h_time = self.dipole(feature_index)
        
        if only_dipole == True:
            return self.out_activation(lstm_out), total_time
        else:
            lstm_out = self.Wlstm(lstm_out)
            torch.cuda.synchronize()
            time_start = time.time()
            output = self.pkgat(neighbor_index, rel_index, h_time)
            torch.cuda.synchronize()
            time_end = time.time()
            total_time = time_end - time_start
            final_lstm = p * lstm_out
            output = final_lstm + (1 - p) * output
            return self.out_activation(output), total_time