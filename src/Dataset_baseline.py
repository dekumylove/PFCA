import torch
from torch.utils.data import TensorDataset

class DiseasePredDataset(TensorDataset):
    def __init__(self, p, p2c, features, rel_index, neighbor_index, feat_index, targets):
        super().__init__()
        self.data = features
        self.p = p
        self.p2c = p2c
        self.rel_index = rel_index
        self.neighbor_index = neighbor_index
        self.feat_index = feat_index

        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        features = self.data[index]
        features = torch.cat(features, dim=0)
        
        p = self.p[index]
        p2c = self.p2c[index]
        rel_index = self.rel_index[index]
        feat_index = self.feat_index[index]
        neighbor_index = self.neighbor_index[index]

        y = self.targets[index]
        y = y.squeeze(dim=0)
        return p, p2c, features, rel_index, neighbor_index, feat_index, y