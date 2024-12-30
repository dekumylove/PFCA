import torch
from torch import nn
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

        return emb


class GraphAttentionLayer(nn.Module):

    def __init__(
        self,
        ninfeat,
        noutfeat,
        dropout,
        alpha,
        hidden_dim,
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
                self.W.append(nn.Parameter(torch.ones(size=(hidden_dim, hidden_dim)), requires_grad = False))   # padding matrixs

        self.linear = nn.Linear(2 * noutfeat, 1, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.device = device
        self.num_rel = nrelation
        self.hidden_dim = hidden_dim
        self.num_feat = ninfeat
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
            nn.Linear(hidden_dim, 1)
        )

    def path_calculation_filtering(self, path_index, path_structure, path_target, feature_embed, h_time, path_filtering):
        """
        This function is used to calculate the path messages and filter the paths.
        :param path_index: extracted paths in the personalized knowledge graphs (PKGs)
        :param path_index: the structure of the extracted paths
        :param path_target: targets connected to each path
        :param feature_embed: feature embeddings tensor(bs, nv, mf, embed_dim)
        :param h_time: patient hidden states obtained from the time-series module
        """
        params = torch.stack([param for param in self.W], dim = 0)
        bs, nv, mf, mp = path_index.size()
        _, np, K, nr = path_structure.size()

        # path calculation
        structures = path_structure.view(-1, nr)
        params = params.view(nr, -1)
        structures = torch.mm(structures, params)
        structures = structures.view(bs, nv, mf, mp, K, self.hidden_dim, self.hidden_dim)
        M_j = torch.prod(structures, dim=-3)  # tensor(bs, nv, mf, mp, self.hidden_dim, self.hidden_dim)
        feat_embed_1 = feature_embed.unsqueeze(dim=-2).repeat(1, 1, 1, mp, 1)
        M_j = torch.einsum('bvfpdd, bvfpd -> bvfpd', M_j, feat_embed_1)

        # path filtering
        h_time_1 = h_time.unsqueeze(dim=-2).unsqueeze(dim=-2).unsqueeze(dim=-2).repeat(1, nv, mf, mp, 1)
        msg = torch.cat([h_time_1, M_j], dim=-1)
        msg = torch.einsum('bvfpa, a -> bvfp', msg, self.W_p)
        mask = F.gumbel_softmax(msg, hard=True)
        path_score = torch.einsum('bvfp, bvfpd -> bvfpd', mask, M_j)
        path_attn = torch.sigmoid(self.path_mlp(path_score)).squeeze(dim=-1)
        if path_filtering:
            M_j = torch.einsum('bvfp, bvfpd -> bvfpd', path_attn, M_j)
        path_target = path_target.view(bs, nv, mf, mp, -1)
        M_j = torch.einsum('bvfpt, bvfpd -> bvftd', path_target, M_j)

        return M_j, path_attn
    
    def joint_impact(self, M_j):
        """
        This function is used to calculate the joint impact of patient features.
        :param M_j: messages transmitted through paths
        """
        bs, nv, mf, nt, hd = M_j.size()
        M_j = M_j.permute(0, 1, 3, 2, 4)
        normalized_embeddings = F.normalize(M_j, p=2, dim=-1)
        normalized_embeddings = normalized_embeddings.reshape(-1, mf, self.hidden_dim)
        normalized_embeddings_2 = normalized_embeddings.reshape(-1, self.hidden_dim, mf)
        cosine_similarity = torch.bmm(normalized_embeddings, normalized_embeddings_2)
        cosine_similarity = cosine_similarity.view(bs, nv, nt, mf, mf)
        M_j = torch.einsum('bvtff, bvtfd -> bvtfd', cosine_similarity, M_j)

        return M_j
    
    def causal_intervention(self, h_time, M_j, intervention = "random_sample"):
        """
        This function is used to conduct the causal intervention.
        :param M_j: messages transmitted through paths
        :param h_time: patient hidden states obtained from the time-series module
        """
        bs, nv, nt, mf, hd = M_j.size()
        h_time_2 = h_time.unsqueeze(dim=-2).unsqueeze(dim=-2).unsqueeze(dim=-2).repeat(1, nv, nt, mf, 1)
        attn_scores = self.causal_mlp(torch.cat([h_time_2, M_j], dim=-1)).squeeze(dim=-1)
        attn_causal = torch.sigmoid(attn_scores - self.causal_threshold)
        attn_trivial = 1 - attn_causal
        g_c = torch.einsum('bvtf, bvtfd -> bvtd', attn_causal, M_j)
        g_c = self.MLP(g_c).squeeze(dim = -1)
        g_t = torch.einsum('bvtf, bvtfd -> bvtd', attn_trivial, M_j)
        g_t = self.MLP(g_t).squeeze(dim = -1)
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

    def forward(self, feature_embed, h_time, path_index, path_structure, path_target, path_filtering, joint_impact, causal_analysis, intervention):
        """
        :param path_index: extracted paths in the personalized knowledge graphs (PKGs)
        :param path_index: the structure of the extracted paths
        :param path_target: targets connected to each path
        :param feature_embed: feature embeddings
        :param h_time: patient hidden states obtained from the time-series module
        """

        # path calculation and filtering
        M_j, path_attn = self.path_calculation_filtering(path_index = path_index, path_structure = path_structure, path_target = path_target, feature_embed = feature_embed, h_time = h_time, path_filtering = path_filtering)

        # joint impact
        if joint_impact:
            M_j = self.joint_impact(M_j = M_j)

        # causal attention
        if causal_analysis:
            g_i, g_c, g_t, attn_causal = self.causal_intervention(h_time = h_time, M_j = M_j, intervention = intervention)

            return g_i, g_c, g_t, path_attn, attn_causal
        else:
            return torch.sum(M_j, dim=-2)


class GATModel(nn.Module):

    def __init__(
        self,
        nfeat,
        nemb,
        gat_layers,
        gat_hid,
        dropout,
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
            threshold=threshold,
            device_id=device_id
        )
        self.output_layer = nn.Sequential(
            nn.Linear(gat_hid, gat_hid),
            nn.Dropout(p=dropout),
            nn.Sigmoid()
        )

    def forward(self, feat_index, path_index, path_structure, path_target, h_time, path_filtering, joint_impact, causal_analysis, intervention = "trivial_mean"):
        """
        :param rel_index: extracted paths in the personalized knowledge graphs (PKGs)
        :param feat_index: patient feature index
        :param h_time: patient hidden states obtained from the time-series module
        """
        # 1. feature embeddings
        embeds = self.embedding.embedding.weight
        feature_embed = torch.einsum('bvmn, nd -> bvmd', feat_index, embeds)

        # 2. Graph Attention Layers
        if causal_analysis:
            g_i, g_c, g_t, path_attentions, causal_attentions = self.pkgat(
                feature_embed=feature_embed,
                h_time=h_time,
                path_index=path_index,
                path_structure=path_structure,
                path_target = path_target,
                path_filtering=path_filtering,
                joint_impact=joint_impact,
                causal_analysis=causal_analysis,
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
        else:
            g_kg = self.self.pkgat(
                feature_embed=feature_embed,
                h_time=h_time,
                path_index=path_index,
                path_structure=path_structure,
                path_filtering=path_filtering,
                joint_impact=joint_impact,
                causal_analysis=causal_analysis,
                intervention=intervention
            )  # (batch_size, num_visit, hidden_dim)
            g_kg = self.output_layer(g_kg)
            g_kg = torch.mean(g_kg, dim=1)  # g_kg: (batch_size, outfeat)
            return g_kg
