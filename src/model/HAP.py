import torch
import numpy as np
import torch.nn.functional as F
from torch import nn as nn


torch.manual_seed(0)
np.random.seed(0)


class AttentionMLP(nn.Module):
    def __init__(self, emb_dim, attn_dim):
        super(AttentionMLP, self).__init__()
        self.emb_dim = emb_dim
        self.attn_dim = attn_dim
        self.Wa = nn.Linear(self.emb_dim * 2, self.attn_dim)
        self.ua = nn.Parameter(torch.randn(self.attn_dim)) 
        self.tanh = nn.Tanh()
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.Wa.weight)
    
    def forward(self, e_i, e_j):
        e_i = e_i.unsqueeze(dim = -2).repeat(1, 1, 1, e_j.size(-2), 1)
        combined_e = torch.cat([e_i, e_j], dim=-1)
        mlp_output = self.tanh(self.Wa(combined_e))
        attn_score = torch.matmul(mlp_output, self.ua)
        return attn_score
    

class HAP(nn.Module):
    def __init__(self, embed_dim, hidden_dim, output_dim, attn_dim = 100, num_feat = 2850
    , dropout_rate=0.1):
        super().__init__()

        self.input_dim = num_feat
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.attn_dim = attn_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.num_feat = num_feat
    
        self.embedding = nn.Embedding(num_embeddings = self.input_dim, embedding_dim = embed_dim)
        
        self.attention_mlp = AttentionMLP(embed_dim, attn_dim)
        self.softmax = nn.Softmax(dim = -1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.out_activation = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, p_, p2c):
        """
        p_: parent features multi-hot vector
        p2c: parent to children features multi-hot vector
        """
        parent_emb = self.embedding(p_.long())
        child_emb = self.embedding(p2c.long())

        # Calculate attention scores
        attn_p2c = self.attention_mlp(parent_emb, child_emb)
        attn_p2c = attn_p2c.to(device=p2c.device)
        attn_c2p = attn_p2c.permute(0, 1, 3, 2)

        attn_p2c = self.softmax(attn_p2c)
        attn_c2p = self.softmax(attn_c2p)

        # up-to-bottom
        G = torch.einsum('bvcf, bvfd -> bvcd', attn_c2p, parent_emb)

        # bottom-to-up
        G = torch.einsum('bvfc, bvcd -> bvfd', attn_p2c, G)

        v_t = torch.sum(G, dim = -2)
        rnn_out, _ = self.rnn(v_t)
        
        output = self.linear(rnn_out)
        output = torch.sum(output, dim = 1)
        
        return self.out_activation(output)