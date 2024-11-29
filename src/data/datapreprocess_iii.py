import pickle
import torch
from tqdm import tqdm
import pandas as pd

with open('../data/mimic-iii/code_map_10.pkl', 'rb') as f:
    code_map = pickle.load(f)

def generate_one_hot():
    """
    generate features and labels for the samples
    """

    with open('../data/mimic-iii/code_map_10.pkl', 'rb') as f:
        code_map = pickle.load(f)

    labels = pd.read_csv('../data/mimic-iii/top_diagnoses.csv')
    labels = [d['Diagnosis'] for _, d in labels.iterrows()]
    diagnoses = pd.read_csv('../data/mimic-iii/diagnoses_icd.csv')
    procedures = pd.read_csv('../data/mimic-iii/procedures_icd.csv')
    prescriptions = pd.read_csv('../data/mimic-iii/prescriptions.csv')
    diagnoses = diagnoses.groupby(['subject_id','hadm_id'])
    procedures = procedures.groupby(['subject_id','hadm_id'])
    prescriptions = prescriptions.groupby(['subject_id','hadm_id'])

    samples = pd.read_csv('../data/mimic-iii/sample_0.7.csv')
    samples_grouped = samples.groupby(['labels'])

    def generate_sample_one_hot(sample_df, dim):
        features_one_hot = []
        label_one_hot = torch.zeros(size=(1, len(labels)))
        label_df = sample_df.iloc[0]
        label_df_splited = label_df['features'].split('_')
        sample_df = sample_df.iloc[1:]
        sample_df_splited = [row['features'].split('_') for _, row in sample_df.iterrows()]

        # generate label's one-hot
        id = (int(label_df_splited[0]), int(label_df_splited[1]))
        pl_diagnoses = diagnoses.get_group(id)
        for _,row in pl_diagnoses.iterrows():
            if row['icd_code'] in labels:
                index = labels.index(row['icd_code'])
                label_one_hot[0][index] = 1

        # generate one-hot of each window as feature
        for i in range(6):
            window_one_hot = torch.zeros(size=(1,dim))
            for sample in sample_df_splited:
                if int(sample[2]) == i + 1:  # the window of this visit is the window i + 1
                    id = (int(sample[0]), int(sample[1]))
                    if id in diagnoses.groups:
                        p_diagnoses = diagnoses.get_group(id)
                        for _,row in p_diagnoses.iterrows():
                            if row['icd_code'] in code_map:
                                window_one_hot[0][code_map[row['icd_code']]] += 1

                    if id in procedures.groups:
                        p_procedures = procedures.get_group(id)
                        for _,row in p_procedures.iterrows():
                            if row['icd_code'] in code_map:
                                window_one_hot[0][code_map[row['icd_code']]] += 1

                    if id in prescriptions.groups:
                        p_prescriptions = prescriptions.get_group(id)
                        for _,row in p_prescriptions.iterrows():
                            if row['drug'] in code_map:
                                window_one_hot[0][code_map[row['drug']]] += 1
                    
            # add this window's one-hot to window_one_hot
            features_one_hot.append(window_one_hot)

        return features_one_hot, label_one_hot

def generate_adjacent_list():
    """load adjacent matrix and convert matrix to adjacent list"""
    adj_list = {}
    with open('../data/mimic-iii/adjacent_matrix.pkl', 'rb') as f:
        adj = pickle.load(f)
    for i in tqdm(range(adj.shape[0]), total = adj.shape[0], desc = 'generating adjacent list'):
        adj_list[i] = {}
        for j in range(adj.shape[1]):
            if adj[i][j] != 0:
                adj_list[i][j] = int(adj[i][j].item())

    with open('../data/mimic-iii/adjacent_list.pkl', 'wb') as e:
        pickle.dump(adj_list, e)

def generate_paths(K=3):
    """
    generate path using the adjacent list
    :param K: The length of the path
    """

    features = torch.load('../data/mimic-iii/features_one_hot.pt')
    with open('../data/mimic-iii/adjacent_list.pkl', 'rb') as f:
        adj_list = pickle.load(f)

    top = pd.read_csv('../data/mimic-iii/top_diagnoses.csv')
    icd = top['Diagnosis']
    target_idx = [code_map[i] for i in icd]

    def find_all_paths(start_idx, path=[]):
        path = path + [start_idx]
        paths = []
        if start_idx in target_idx:
            if len(path) == 1:
                paths.append(path)
            else:
                return [path]
        if start_idx not in adj_list or len(path) >= K:
            return []
        
        for node, rel in adj_list[start_idx].items():
            if node not in path:
                new_paths = find_all_paths(node, path)
                for p in new_paths:
                    paths.append(p)
                    if len(paths) > 3:
                        return paths
        return paths

    paths = []

    # generate paths for each sample
    for sample in tqdm(features, total = len(features), desc = 'generating paths'):
        sample_paths = []
        for visit in sample:
            visit_paths = {}
            for i in range(visit.shape[1]):
                if visit[0][i] != 0:
                    all_paths = []
                    all_paths = find_all_paths(i)
                    visit_paths[i] = all_paths
                
            sample_paths.append(visit_paths)
        paths.append(sample_paths)
    with open(f'../data/mimic-iii/paths_{K}.pkl', 'wb') as e:
        pickle.dump(paths, e)

def generate_rel_index(num_feat = 2850, max_feat = 12, num_rel = 12, K = 4, num_path = 8, max_target = 12):
    """
    generate relation index and feat index
    :param num_feat:          The number of the medical feature
    :param max_feat:          The maximum number of the patient feature in a visit
    :param num_rel:           The number of the relation types
    :param K:                 The length of the path
    :param num_path:          The maximum number of the paths linked with a feature
    :param max_target:        The maximum number of the target linked with a feature
    """

    with open(f'../data/mimic-iii/paths_{K}.pkl', 'rb') as f:
        paths3 = pickle.load(f)
    with open('../data/mimic-iii/adjacent_matrix.pkl', 'rb') as f:
        adj = pickle.load(f)
    rel_index = []
    feat_index = []
    paths = []

    # generate data for each sample
    for sample in tqdm(paths3, total = len(paths3)):
        sample_rel = []
        sample_feat = []
        sample_path = []
        for visit in sample:
            visit_rel = torch.zeros(size=(max_feat, max_target, num_path, K, num_rel))
            visit_feat = []
            real_feat = torch.zeros(size=(max_feat, num_feat))
            visit_path = [[] for i in range(max_feat)]
            for k, v in visit.items():
                ok = k
                if k not in visit_feat:
                    if len(visit_feat) >= max_feat:
                        break
                    visit_feat.append(k)
                k = visit_feat.index(k)
                real_feat[k][ok] = 1
                target_ = []

                # traverse the paths and convert them into indexes
                for path_idx, path in enumerate(v):
                    if path_idx < num_path:
                        target = path[-1]
                        if target not in target_:
                            target_.append(target)
                            if len(target_) > max_target:
                                break
                        target = target_.index(target)
                        for path_index in range(num_path):
                            slice_tensor = visit_rel[:, :, path_index, :, :]
                            if torch.sum(slice_tensor).item() == 0:
                                visit_path[k].append(path)
                                for i in range(len(path) - 1):
                                    visit_rel[k][target][path_index][i][int(adj[path[i]][path[i+1]])] = 1
                                if len(path) - 1 < K:
                                    for j in range(K - len(path) + 1):
                                        visit_rel[k][target][path_index][len(path) - 1 + j][0] = 1
                                break
                    else:
                        break
            sample_feat.append(real_feat)
            sample_rel.append(visit_rel)
            sample_path.append(visit_path)
        sample_feat = torch.stack(sample_feat, dim=0)
        feat_index.append(sample_feat)
        sample_rel = torch.stack(sample_rel, dim=0)
        rel_index.append(sample_rel)
        paths.append(sample_path)

    torch.save(rel_index, f'../data/mimic-iii/rel_index_{K}.pt')
    torch.save(feat_index, f'../data/mimic-iii/feat_index_{K}.pt')             

def filter_label():
    """filter label that is imbalanced to measure the metrics"""

    labels = torch.load('../data/mimic-iii/label_one_hot.pt')
    labels_C = torch.cat(labels, dim = 0)
    labels_C = torch.sum(labels_C, dim = 0)
    values, indices = torch.topk(labels_C, 10, largest=False)

    top = pd.read_csv('../data/mimic-iii/top_diagnoses.csv')
    icd = top['Diagnosis']
    target_idx = [code_map[i] for i in icd]
    filter_idx = indices.tolist()
    target = [item for i, item in enumerate(target_idx) if i not in filter_idx]

    return target

def generate_knowledge_driven_data(max_feat = 10, num_feat = 2850, max_target = 4, num_rel = 11):
    """
    generate graphcare, har, medpath data
    :param num_feat:          The number of the medical feature
    :param max_feat:          The maximum number of the patient feature in a visit
    :param num_rel:           The number of the relation types
    :param max_target:        The maximum number of the target linked with a feature
    """
    
    with open('../data/mimic-iii/adjacent_matrix.pkl', 'rb') as f:
        adj = pickle.load(f)
    features = torch.load('../data/mimic-iii/features_one_hot.pt')

    feat_index = []
    rel_index = []
    neighbor_index = []

    # generate for each sample
    for sample in tqdm(features, total=len(features), desc="generating knowledge_driven_data"):
        sample_f_index = []
        sample_n_index = []
        sample_r_index = []
        for visit in sample:
            visit_f_index = torch.zeros(size=(max_feat, num_feat))
            visit_n_index = torch.zeros(size=(max_feat, max_target, num_feat))
            visit_r_index = torch.zeros(size=(max_feat, max_target, num_rel))
            feat_num = 0
            for i in range(visit.shape[1]):
                if visit[0][i] != 0:
                    visit_f_index[feat_num][i] = 1
                    target_num = 0
                    for j in range(adj.shape[1]):
                        if adj[i][j] != 0:
                            visit_n_index[feat_num][target_num][j] = 1
                            visit_r_index[feat_num][target_num][int(adj[i][j]) - 1] = 1
                            target_num += 1
                            if target_num >= max_target:
                                break
                    feat_num += 1
                    if feat_num >= max_feat:
                        break
            sample_f_index.append(visit_f_index)
            sample_n_index.append(visit_n_index)
            sample_r_index.append(visit_r_index)
        sample_f_index = torch.stack(sample_f_index, dim=0)
        sample_n_index = torch.stack(sample_n_index, dim=0)
        sample_r_index = torch.stack(sample_r_index, dim=0)
        feat_index.append(sample_f_index)
        rel_index.append(sample_r_index)
        neighbor_index.append(sample_n_index)

    with open('../data/mimic-iii/feat_index.pkl', 'wb') as f:
        pickle.dump(feat_index, f)
    with open('../data/mimic-iii/rel_index.pkl', 'wb') as f:
        pickle.dump(rel_index, f)
    with open('../data/mimic-iii/neighbor_index.pkl', 'wb') as f:
        pickle.dump(neighbor_index, f)

def generate_drugrecommendation_data():
    """generate_drugrecommendation_data"""
    
    features = torch.load('../data/mimic-iii/features_one_hot.pt')
    drug_count = {}
    for sample in features:
        for visit in sample:
            for i in range(1164):
                if visit[0][1686+i] != 0:
                    k = 1686+i
                    if k not in drug_count:
                        drug_count[k] = 1
                    else:
                        drug_count[k] += 1
    sorted_items = sorted(drug_count.items(), key=lambda x: x[1], reverse=True)
    sorted_items = sorted_items[:90]
    keys = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    drugrecommendation_features_one_hot = []
    drugrecommendation_label_one_hot = []

    # generate data for each sample
    for sample in tqdm(features, total=len(features), desc="generating drugrecommendation data"):
        sample_feature = []
        label = torch.zeros(size=(1, 90))
        for visit_index, visit in enumerate(sample):
            if visit_index == len(sample) - 1:
                visit_feature = visit.clone()
                for i in range(1164):   # 1164 is the number of the medication features
                    k = i + 1686    # 1686 is the index of the first medication feature
                    if visit[0][k] != 0 and k in values:
                        label[0][values.index(k)] = 1
                        visit_feature[0][k] = 0
                sample_feature.append(visit_feature)
            else:
                sample_feature.append(visit)
        drugrecommendation_features_one_hot.append(sample_feature)
        drugrecommendation_label_one_hot.append(label)
    torch.save(drugrecommendation_features_one_hot, '../data/mimic-iii/drugrecommendation_features_one_hot.pt')
    torch.save(drugrecommendation_label_one_hot, '../data/mimic-iii/drugrecommendation_label_one_hot.pt')

#----------------------------------------------------------------------generate_readmission_data-----------------------------------------------------------------------------------------------

def generate_readmission_paths():
    """generate_readmission_paths"""
    K = 3 #path's length
    features = torch.load('readmission_features_one_hot.pt')
    with open('adjacent_list.pkl', 'rb') as f:
        adj_list = pickle.load(f)

    def find_all_paths(start_idx, target_idx, path=[]):
        path = path + [start_idx]
        paths = []
        if start_idx in target_idx:
            if len(path) == 1:
                paths.append(path)
            else:
                return [path]
        if start_idx not in adj_list or len(path) >= K:
            return []
        for node, rel in adj_list[start_idx].items():
            if node not in path:
                new_paths = find_all_paths(node, target_idx, path)
                for p in new_paths:
                    paths.append(p)
                    if len(paths) > K:
                        return paths
        return paths

    paths = []
    for sample in tqdm(features, total = len(features), desc = 'generating readmission paths'):
        sample_paths = []
        for visit in sample:
            target_idx = [i for i in range(visit.shape[1]) if visit[0][i] != 0]
            visit_paths = {}
            for i in range(visit.shape[1]):
                if visit[0][i] != 0:
                    all_paths = []
                    all_paths = find_all_paths(i, target_idx)
                    visit_paths[i] = all_paths
            sample_paths.append(visit_paths)
        paths.append(sample_paths)
    with open(f'readmission_paths_{K}.pkl', 'wb') as e:
        pickle.dump(paths, e)

def generate_readmission_rel_index(num_feat = 2850, max_feat = 16, num_rel = 12, K = 3, num_path = 12, num_target = 90):
    """
    generate relation index for readmission
    :param num_feat:          The number of the medical feature
    :param max_feat:          The maximum number of the patient feature in a visit
    :param num_rel:           The number of the relation types
    :param num_target:        The maximum number of the target linked with a feature
    :param num_path:          The maximum number of paths linked with a feature
    """
    with open(f'readmission_paths_{K}.pkl', 'rb') as f:
        paths3 = pickle.load(f)
    with open('adjacent_matrix.pkl', 'rb') as f:
        adj = pickle.load(f)
    rel_index = []
    feat_index = []
    for sample in tqdm(paths3, total = len(paths3)):
        sample_rel = []
        sample_feat = []
        for visit in sample:
            visit_rel = torch.zeros(size=(max_feat, num_target, num_path, K, num_rel))
            visit_feat = []
            real_feat = torch.zeros(size=(max_feat, num_feat))
            for k, v in visit.items():
                ok = k
                if k not in visit_feat:
                    if len(visit_feat) >= max_feat:
                        break
                    visit_feat.append(k)
                k = visit_feat.index(k)
                real_feat[k][ok] = 1
                target_ = []
                for path_idx, path in enumerate(v):
                    if path_idx < num_path:
                        target = path[-1]
                        if target not in target_:
                            target_.append(target)
                            if len(target_) > num_target:
                                break
                        target = target_.index(target)
                        for path_index in range(num_path):
                            slice_tensor = visit_rel[:, :, path_index, :, :]
                            if torch.sum(slice_tensor).item() == 0:
                                for i in range(len(path) - 1):
                                    visit_rel[k][target][path_index][i][int(adj[path[i]][path[i+1]])] = 1
                                if len(path) - 1 < K:
                                    for j in range(K - len(path) + 1):
                                        visit_rel[k][target][path_index][len(path) - 1 + j][0] = 1
                                break
                    else:
                        break
            sample_feat.append(real_feat)
            sample_rel.append(visit_rel)
        sample_feat = torch.stack(sample_feat, dim=0)
        feat_index.append(sample_feat)
        sample_rel = torch.stack(sample_rel, dim=0)
        rel_index.append(sample_rel)
    print(len(rel_index))
    print(len(feat_index))
    torch.save(rel_index, f'readmission_rel_index_{K}.pt')
    torch.save(feat_index, f'readmission_feat_index_{K}.pt')