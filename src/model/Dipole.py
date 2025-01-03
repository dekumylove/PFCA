import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.nn.parameter import Parameter

"""
This module implements three variants of the Dipole attention mechanism for sequential data:
- Location-based attention (Dip_l)
- General attention (Dip_g)
- Concatenation-based attention (Dip_c)

Each model uses a GRU as the base encoder and implements different attention mechanisms
for capturing temporal dependencies in the input sequence.
"""

class Dip_l(nn.Module):
    """
    Location-based Attention Dipole model.
    This model learns attention weights based purely on the hidden state positions,
    without considering the hidden state content.
    """
    def __init__(self, input_dim, output_dim, hidden_dim, bi_direction=False, device=torch.device('cuda'), activation='sigmoid'):
        super(Dip_l, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.bi_direction = bi_direction

        self.bi = 1
        if self.bi_direction:
            self.bi = 2
        self.rnn = nn.GRU(input_dim, self.hidden_dim, 1, bidirectional=self.bi_direction)

        self.attention_t = nn.Linear(self.hidden_dim * self.bi, 1)
        self.a_t_softmax = nn.Softmax(dim=1)

        self.linear1 = nn.Linear(self.hidden_dim * 2 * self.bi, self.output_dim)
        self.linear2 = nn.Linear(self.hidden_dim * self.bi, self.output_dim)
        if activation == 'sigmoid':
            self.out_activation = nn.Sigmoid()
        else:
            self.out_activation = nn.Softmax(1)

    def forward(self, x, verbose=False):
        self.rnn.flatten_parameters()
        rnn_in = x.permute(1, 0, 2)
        rnn_out, _ = self.rnn(rnn_in)
        rnn_out = rnn_out.permute(1, 0, 2)
        out = rnn_out[:, :-1, :]
        if verbose:
            print("out", out.shape, '\n', out)

        location_attention = self.attention_t(out)
        a_t_softmax_out = self.a_t_softmax(location_attention)
        context = torch.mul(a_t_softmax_out, out)
        sum_context = context.sum(dim=1)

        if verbose:
            print("location_attention", location_attention.shape, '\n', location_attention)
            print("a_t_softmax_out", a_t_softmax_out.shape, '\n', a_t_softmax_out)
            print("context", context.shape, '\n', context)

        concat_input = torch.cat([rnn_out[:, -1, :], sum_context], 1)
        output = self.linear1(concat_input)
        output = self.out_activation(output)
        return output, self.linear2(rnn_out)


class Dip_g(nn.Module):
    """
    General Attention Dipole model.
    This model learns attention weights using a learned linear transformation
    between the query (last hidden state) and keys (all hidden states).
    """
    def __init__(self, input_dim, output_dim, hidden_dim, bi_direction=False, device=torch.device('cuda'), activation='sigmoid'):
        super(Dip_g, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.bi_direction = bi_direction

        self.bi = 1
        if self.bi_direction:
            self.bi = 2
        self.rnn = nn.GRU(input_dim, self.hidden_dim, 1, bidirectional=self.bi_direction)

        self.attention_t_w = Parameter(
            torch.randn(self.hidden_dim * self.bi, self.hidden_dim * self.bi, requires_grad=True).float())
        self.a_t_softmax = nn.Softmax(dim=1)

        self.linear1 = nn.Linear(self.hidden_dim * 2 * self.bi, self.output_dim)
        self.linear2 = nn.Linear(self.hidden_dim * self.bi, self.output_dim)
        if activation == 'sigmoid':
            self.out_activation = nn.Sigmoid()
        else:
            self.out_activation = nn.Softmax(1)

    def forward(self, x, verbose=False):
        self.rnn.flatten_parameters()
        rnn_in = x.permute(1, 0, 2)
        rnn_out, _ = self.rnn(rnn_in)
        rnn_out = rnn_out.permute(1, 0, 2)
        out = rnn_out[:, :-1, :]
        last_out = rnn_out[:, -1, :]
        if verbose:
            print("out", out.shape, '\n', out)

        general_attention = torch.matmul(last_out, self.attention_t_w)
        general_attention = torch.matmul(out, general_attention.unsqueeze(-1))
        a_t_softmax_out = self.a_t_softmax(general_attention)
        context = torch.mul(a_t_softmax_out, out)
        sum_context = context.sum(dim=1)

        if verbose:
            print("general_attention", general_attention.shape, '\n', general_attention)
            print("a_t_softmax_out", a_t_softmax_out.shape, '\n', a_t_softmax_out)
            print("context", context.shape, '\n', context)

        concat_input = torch.cat([last_out, sum_context], 1)

        output = self.linear1(concat_input)
        output = self.out_activation(output)
        return output, self.linear2(rnn_out)


class Dip_c(nn.Module):
    """
    Concatenation-based Attention Dipole model.
    This model learns attention weights by concatenating each hidden state
    with the last hidden state, followed by a non-linear transformation.
    """
    def __init__(self, input_dim, output_dim, hidden_dim, max_timesteps, bi_direction=False, device=torch.device('cuda'), activation='sigmoid'):
        super(Dip_c, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.bi_direction = bi_direction
        self.max_timesteps = max_timesteps

        self.bi = 1
        if self.bi_direction:
            self.bi = 2
        self.rnn = nn.GRU(input_dim, self.hidden_dim, 1, bidirectional=self.bi_direction)

        self.latent = 16
        self.attention_t = nn.Linear(self.hidden_dim * 2 * self.bi, self.latent, bias=False)
        self.attention_v = nn.Linear(self.latent, 1, bias=False)
        self.a_t_softmax = nn.Softmax(dim=1)

        self.linear1 = nn.Linear(self.hidden_dim * 2 * self.bi, self.output_dim)
        self.linear2 = nn.Linear(self.hidden_dim * self.bi, self.hidden_dim)
        if activation == 'sigmoid':
            self.out_activation = nn.Sigmoid()
        else:
            self.out_activation = nn.Softmax(1)

    def forward(self, x, verbose=False):
        self.rnn.flatten_parameters()
        rnn_in = x.permute(1, 0, 2)
        rnn_out, _ = self.rnn(rnn_in)
        rnn_out = rnn_out.permute(1, 0, 2)
        out = rnn_out[:, :-1, :]
        last_out = rnn_out[:, -1, :]

        re_ht = last_out.unsqueeze(1).repeat(1, self.max_timesteps - 1, 1)
        concat_input = torch.cat([re_ht, out], 2)
        concatenation_attention = self.attention_t(concat_input)
        concatenation_attention = torch.tanh(concatenation_attention)
        concatenation_attention = self.attention_v(concatenation_attention)
        a_t_softmax_out = self.a_t_softmax(concatenation_attention)
        self.a_t_softmax_out = a_t_softmax_out
        context = torch.mul(a_t_softmax_out, out)
        sum_context = context.sum(dim=1)

        if verbose:
            print("re_ht", re_ht.shape, '\n', re_ht)
            print("concat_input", concat_input.shape, '\n', concat_input)
            print("concatenation_attention", concatenation_attention.shape, '\n', concatenation_attention)
            print("a_t_softmax_out", a_t_softmax_out.shape, '\n', a_t_softmax_out)
            print("context", context.shape, '\n', context)
            print("sum_context", sum_context.shape, '\n', sum_context)

        concat_input = torch.cat([last_out, sum_context], 1)

        output = self.linear1(concat_input)
        output = self.out_activation(output)
        return output, self.linear2(rnn_out)