import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
import pickle


def del_tensor_ele(arr: torch.tensor, index):
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return torch.cat((arr1, arr2), dim=0)


def read_data(feature_file, label_file, device):
    with open(feature_file, 'rb') as f:
        features = pickle.load(f)
    labels = torch.load(label_file)
    
    sum_tensor = torch.sum(torch.stack(labels).squeeze(dim=1), dim=0).to(device)
    
    nums_disease = {i:num for i, num in enumerate(sum_tensor.tolist())}
    nums_disease = dict(sorted(nums_disease.items(), key=lambda item: item[1]))
    delete_index = list(nums_disease.keys())[0:10]
    new_labels = []
    for i in tqdm(labels, total=len(labels)):
        i.to(device)
        label = torch.squeeze(i, dim=0)
        for i, idx in enumerate(delete_index):
            label = del_tensor_ele(label, idx-i)
        new_labels.append(label)
    return features, new_labels


class DiseasePredDataset(TensorDataset):
    def __init__(self, features, feat_index, path_index, path_structure, path_target, targets):
        super().__init__()
        self.data = features
        self.feat_index = feat_index
        self.path_index = path_index
        self.path_structure = path_structure
        self.path_target = path_target
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        features = self.data
        features = features[index]
        features = torch.cat(features, dim=0)
        feat_index = self.feat_index
        feat_index = feat_index[index]
        path_index = self.path_index
        path_index = path_index[index]

        path_structure = self.path_structure
        path_structure = path_structure[index]

        path_target = self.path_target
        path_target = path_target[index]

        y = self.targets[index]
        y = y.squeeze(dim=0)
        return features, feat_index, path_index, path_structure, path_target, y
