import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.nn.parameter import Parameter


class LSTM_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, bi_direction=False, activation="sigmoid"):
        super(LSTM_Model, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.bi_direction = bi_direction
        self.bi = 1
        if self.bi_direction:
            self.bi = 2

        self.LSTM = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=self.bi_direction
        )

        self.output = nn.Linear(self.hidden_dim * self.bi, self.output_dim)
        self.linear2 = nn.Linear(self.hidden_dim * self.bi, self.output_dim)
        
        if activation == 'sigmoid':
            self.out_activation = nn.Sigmoid()
        else:
            self.out_activation = nn.Softmax(1)
    
    def forward(self, x):
        """
        :param x: (batch_size, num_visit, num_feat)
        """
        self.LSTM.flatten_parameters()

        lstm_in = x  
        lstm_out, _ = self.LSTM(lstm_in)

        c = torch.sum(lstm_out, dim=1)
        output = self.output(c)
        return output, self.linear2(c)