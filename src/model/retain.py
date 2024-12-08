import torch
import torch.nn.functional as F
from torch import nn as nn

class Retain(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, bi_direction=False, device=torch.device('cuda'),
                 activation="sigmoid"):
        super(Retain, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = input_dim
        self.bi_direction = bi_direction
        self.bi = 1
        if self.bi_direction:
            self.bi = 2
        self.alpha_gru = nn.GRU(
            self.input_dim,
            self.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=self.bi_direction
        )
        self.beta_gru = nn.GRU(
            self.input_dim,
            self.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=self.bi_direction
        )

        self.alpha_li = nn.Linear(self.hidden_dim * self.bi, 1)
        self.beta_li = nn.Linear(self.hidden_dim * self.bi, self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim * self.bi, self.output_dim)
        self.linear2 = nn.Linear(self.hidden_dim * self.bi, self.output_dim)
        if activation == 'sigmoid':
            self.out_activation = nn.Sigmoid()
        else:
            self.out_activation = nn.Softmax(1)

    def forward(self, x):
        self.alpha_gru.flatten_parameters()
        self.beta_gru.flatten_parameters()

        rnn_in = x

        # visit-level attention
        g, _ = self.alpha_gru(rnn_in)
        # feature-level attention
        h, _ = self.beta_gru(rnn_in)

        g_li = self.alpha_li(g)
        h_li = self.beta_li(h)

        attn_g = F.softmax(g_li, dim=-1)
        attn_h = torch.tanh(h_li)

        c = attn_g * attn_h * (x)
        c = torch.sum(c, dim=1)

        output = self.output(c)
        return output, self.linear2(g)