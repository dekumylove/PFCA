import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self, num_feat, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(
            num_feat, hidden_dim
        )
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, x):
        emb = self.embedding(x)
        return emb


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer for processing medical pathways.
    Implements multi-relational attention mechanism over patient records.
    """
    def __init__(
        self,
        ninfeat,
        noutfeat,
        alpha,
        hidden_dim,
        dropout=0.5,
        nrelation=12,
        device=torch.device("cuda"),
    ):
        super().__init__()

        # Initialize relation-specific transformation matrices
        self.W = nn.ParameterList()

        for i in range(nrelation):
            if i != 0:
                self.W.append(nn.Parameter(torch.rand(size=(hidden_dim, hidden_dim))))
                nn.init.xavier_uniform_(self.W[-1].data, gain=1.414)
            else:
                self.W.append(nn.Parameter(torch.ones(size=(hidden_dim, hidden_dim)), 
                                         requires_grad=False))

        self.linear = nn.Linear(2 * noutfeat, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.device = device
        self.num_rel = nrelation
        self.hidden_dim = hidden_dim
        self.num_feat = ninfeat

        # Feature transformation layers
        self.Linear_x = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.Linear_i = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.Linear_y = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.softmax = nn.Softmax(dim=-1)
        self.Linear = nn.Linear(hidden_dim + noutfeat, 1)
        self.output_layer = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, noutfeat),
            nn.ReLU()
        )

    def forward(self, source_embed, rel_index, s):
        """
        :param rel_index: extracted paths in the personalized knowledge graphs (PKGs)
        :param source_embed: feature embeddings
        :param s: patient hidden states obtained from the time-series module
        :return:        FloatTensor B*F*(headxE2)
        """
        # Stack relation transformation matrices
        params = [param for param in self.W]
        params = torch.stack(params, dim=0)
        params = params.view(self.num_rel, -1)
        bs, nv, mf, mt, np, K, nr = rel_index.size()

        # Calculate path representations
        rels = rel_index.view(-1, nr)
        rels = torch.mm(rels, params)
        rels = rels.view(bs, nv, mf, mt, np, K, self.hidden_dim, self.hidden_dim)
        W = torch.prod(rels, dim=-3)

        # Apply attention mechanism
        z_j = W.permute(0, 1, 3, 2, 4, 5, 6)
        z_j = torch.einsum('bvtfpdd, bvtfd -> bvtfpd', z_j, source_embed)
        s = s.unsqueeze(dim = -2).unsqueeze(dim = -2).unsqueeze(dim = -2).repeat(1, 1, mt, mf, np, 1)
        attn = self.Linear(torch.cat([s, z_j], dim=-1)).squeeze(dim=-1)
        attn = self.softmax(attn)
        z_j = torch.einsum('bvtfp, bvtfpd -> bvtfd', attn, z_j)
        
        # Generate final output
        output = torch.sum(self.output_layer(z_j), dim=-2)
        output = torch.sum(output, dim=-2)

        return output


class MedPath(nn.Module):
    """
    Medical Pathway model that combines embedding and graph attention
    for processing patient records and predicting medical outcomes.
    """
    def __init__(
        self,
        nfeat,
        nemb,
        gat_layers,
        gat_hid,
        alpha=0.2,
        nrelation=12,
        device=torch.device("cuda")
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
            alpha=alpha,
            nrelation=self.nrelation,
            device=self.device,
            hidden_dim=self.hidden_dim
        )

    def forward(self, neighbor_index, rel_index, h_time):
        """
        :param feature_index : patient multi-hot EHR records in the PKGs
        :param neighbor_index: prediction targets multi-hot vector
        :param rel_index: extracted paths in the personalized knowledge graphs (PKGs)
        """
        # Generate embeddings
        embeds = self.embedding.embedding.weight
        source_embed = torch.einsum('bvtmn, nd -> bvtmd', neighbor_index.permute(0, 1, 3, 2, 4), embeds)

        # Process through graph attention
        output = self.pkgat(source_embed=source_embed, rel_index=rel_index, s=h_time)
        output = torch.mean(output, dim=1)

        return output