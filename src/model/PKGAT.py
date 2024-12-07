import torch
from torch import nn
import time
import torch.nn.functional as F


class Embedding(nn.Module):
    def __init__(self, num_feat, hidden_dim):
        """
        :param num_feat: number of features
        :param hidden_dim: embedding size
        """
        super().__init__()
        self.embedding = nn.Embedding(
            num_feat, hidden_dim
        )
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, x):
        emb = self.embedding(x)

        return emb  # F*E


class GraphAttentionLayer(nn.Module):

    def __init__(
        self,
        ninfeat,
        noutfeat,
        dropout,
        alpha,
        hidden_dim,
        num_path,
        threshold,
        nrelation=12,
        device_id=1,
        device=torch.device("cuda"),
    ):
        super().__init__()

        self.W = nn.ParameterList()

        for i in range(nrelation):
            if i != 0:
                self.W.append(nn.Parameter(torch.zeros(size=(hidden_dim, hidden_dim))))
                nn.init.xavier_uniform_(self.W[-1].data, gain=1.414)
            else:
                self.W.append(nn.Parameter(torch.ones(size=(hidden_dim, hidden_dim)), requires_grad = False))

        self.linear = nn.Linear(2 * noutfeat, 1, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.device = device
        self.num_rel = nrelation
        self.num_path = num_path
        self.hidden_dim = hidden_dim
        self.num_feat = ninfeat  # 特征的数量
        self.W_p = nn.Parameter(torch.ones(hidden_dim+noutfeat))
        self.b_p = nn.Parameter(torch.ones(1))
        self.sigmoid = nn.Sigmoid()
        self.threshold = threshold
        self.device_id = device_id
        self.softmax = nn.Softmax(dim = -1)
        self.causal_threshold = nn.Parameter(torch.tensor(0.5))
        self.path_threshold = nn.Parameter(torch.tensor(0.01))
        self.causal_mlp = nn.Linear(hidden_dim + noutfeat, 1)
        self.path_mlp = nn.Linear(hidden_dim, 1)
        self.MLP = nn.Sequential(
            nn.Linear(hidden_dim+noutfeat, self.num_feat),
            nn.ReLU(),
            nn.Linear(self.num_feat, 1)
        )

    def path_calculation_filtering(self, rel_index, feature_embed, h_time):
        """
        This function is used to calculate the path messages and filter the paths.
        :param rel_index: extracted paths in the personalized knowledge graphs (PKGs)
        :param feature_embed: feature embeddings
        :param h_time: patient hidden states obtained from the time-series module
        """
        params = torch.stack([param for param in self.W], dim = 0)
        bs, nv, mf, mt, np, K, nr = rel_index.size()

        # path calculation
        rels = rel_index.view(-1, nr)
        rels = torch.mm(rels, params)
        rels = rels.view(bs, nv, mf, mt, np, K, self.hidden_dim, self.hidden_dim)
        M_j = torch.prod(rels, dim=-3)

        # path filtering
        feat_embed_1 = feature_embed.unsqueeze(dim=-2).unsqueeze(dim=-2).repeat(1, 1, 1, mt, np, 1)
        M_j = torch.einsum('bvftpdd, bvftpd -> bvftpd', M_j, feat_embed_1)
        h_time_1 = h_time.unsqueeze(dim=-2).unsqueeze(dim=-2).unsqueeze(dim=-2).repeat(1, 1, mf, mt, np, 1)
        msg = torch.cat([h_time_1, M_j], dim=-1)
        msg = torch.einsum('bvftpa, a -> bvftp', msg, self.W_p)
        mask = F.gumbel_softmax(msg, hard=True)
        path_score = torch.einsum('bvftp, bvftpd -> bvftpd', mask, M_j)
        path_attn = torch.sigmoid(self.path_mlp(path_score)).squeeze(dim=-1)
        M_j = torch.einsum('bvftp, bvftpd -> bvftd', path_attn, M_j)
        M_j = torch.sum(M_j, dim=-2)

        return M_j, path_attn
    
    def joint_impact(self, M_j):
        """
        This function is used to calculate the joint impact of patient features.
        :param M_j: messages transmitted through paths
        """
        bs, nv, mf, hd = M_j.size()
        normalized_embeddings = F.normalize(M_j, p=2, dim=-1)
        normalized_embeddings = normalized_embeddings.view(-1, mf, self.hidden_dim)
        normalized_embeddings_2 = normalized_embeddings.view(-1, self.hidden_dim, mf)
        cosine_similarity = torch.bmm(normalized_embeddings, normalized_embeddings_2)
        cosine_similarity = cosine_similarity.view(bs, nv, mf, mf)
        M_j = torch.einsum('bvff, bvfd -> bvfd', cosine_similarity, M_j)

        return M_j
    
    def causal_intervention(self, h_time, M_j, intervention = "random_sample"):
        """
        This function is used to conduct the causal intervention.
        :param M_j: messages transmitted through paths
        :param h_time: patient hidden states obtained from the time-series module
        """
        h_time_2 = h_time.unsqueeze(dim=-2).repeat(1, 1, M_j.size(2), 1)
        attn_scores = self.causal_mlp(torch.cat([h_time_2, M_j], dim=-1)).squeeze(dim=-1)
        attn_causal = torch.sigmoid(attn_scores - self.causal_threshold)
        attn_trivial = 1 - attn_causal
        g_c = torch.einsum('bvf, bvfd -> bvd', attn_causal, M_j)
        g_t = torch.einsum('bvf, bvfd -> bvd', attn_trivial, M_j)
        g_i = g_c.clone()

        if intervention == "random_sample":
            M = g_c.size(0)
            II = torch.randint(0, M, (M,), device=g_c.device)
            II = II + (II == torch.arange(M, device=g_c.device)).int()
            g_i += g_t[II]
        elif intervention == "random_visit":
            M = g_c.size(0)
            seq_len = g_c.size(1)
            II = torch.randint(0, seq_len, (M,), device=g_c.device)
            g_i[torch.arange(M), II] += g_t[torch.arange(M), II]

        return g_i, g_c, g_t, attn_causal

    def forward(self, feature_embed, h_time, rel_index, intervention):
        """
        :param rel_index: extracted paths in the personalized knowledge graphs (PKGs)
        :param feature_embed: feature embeddings
        :param h_time: patient hidden states obtained from the time-series module
        """

        # path calculation and filtering
        M_j, path_attn = self.path_calculation_filtering(rel_index = rel_index, feature_embed = feature_embed, h_time = h_time)

        # joint impact
        M_j = self.joint_impact(M_j = M_j)

        # causal attention
        g_i, g_c, g_t, attn_causal = self.causal_intervention(h_time = h_time, M_j = M_j, intervention = intervention)

        return g_i, g_c, g_t, path_attn, attn_causal


class GATModel(nn.Module):

    def __init__(
        self,
        nfeat,
        nemb,
        gat_layers,
        gat_hid,
        dropout,
        num_path,
        threshold,
        alpha=0.2,
        nrelation=12,
        device_id=1,
        device=torch.device("cuda"),
    ):
        super().__init__()

        self.embedding = Embedding(nfeat, nemb)
        self.gat_layers = gat_layers
        self.gats = torch.nn.ModuleList()
        self.nrelation = nrelation
        self.device = device
        self.hidden_dim = nemb
        ninfeat = nemb
        self.pkgat = GraphAttentionLayer(
            ninfeat=ninfeat,
            noutfeat=gat_hid,
            dropout=dropout,
            alpha=alpha,
            nrelation=self.nrelation,
            device=self.device,
            hidden_dim=self.hidden_dim,
            num_path=num_path,
            threshold=threshold,
            device_id=device_id
        )
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, gat_hid),
            nn.Dropout(p=dropout),
            nn.Sigmoid()
        )

    def forward(self, rel_index, feat_index, h_time, intervention = "random_sample"):
        """
        :param rel_index: extracted paths in the personalized knowledge graphs (PKGs)
        :param feat_index: patient feature index
        :param h_time: patient hidden states obtained from the time-series module
        """
        # 1. feature embeddings
        embeds = self.embedding.embedding.weight
        feature_embed = torch.einsum('bvmn, nd -> bvmd', feat_index, embeds)

        # 2. Graph Attention Layers
        g_i, g_c, g_t, path_attentions, causal_attentions = self.pkgat(
            feature_embed=feature_embed,
            h_time=h_time,
            rel_index=rel_index,
            intervention=intervention
        )  # (batch_size, num_visit, hidden_dim)

        g_i = self.output_layer(g_i)
        g_c = self.output_layer(g_c)
        g_t = self.output_layer(g_t)
        g_i = torch.mean(g_i, dim=1)  # g_i: (batch_size, outfeat)
        g_c = torch.mean(g_c, dim=1)  # g_c: (batch_size, outfeat)
        g_t = torch.mean(g_t, dim=1)  # g_t: (batch_size, outfeat)
        if intervention == "trivial_mean":
            g_i = g_i + g_t

        return g_i, g_c, g_t, path_attentions, causal_attentions