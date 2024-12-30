import pickle
import torch
from tqdm import tqdm
import pandas as pd
import argparse

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
    
    def generate_one_hot():
        features_one_hot_list = []
        label_one_hot_list = []
        for key, sample in tqdm(samples_grouped, total=len(samples_grouped), desc="Generating samples' one-hot encoddings"):
            features_one_hot, label_one_hot = generate_sample_one_hot(sample, dim=2850)
            features_one_hot_list.append(features_one_hot)
            label_one_hot_list.append(label_one_hot)
        
        return features_one_hot_list, label_one_hot_list

    features_one_hot_list, label_one_hot_list = generate_one_hot()

    print(len(features_one_hot_list), len(label_one_hot_list))
    assert len(features_one_hot_list) == len(label_one_hot_list)

    torch.save(features_one_hot_list, '../data/mimic-iii/features_one_hot.pt')
    torch.save(label_one_hot_list, '../data/mimic-iii/label_one_hot.pt')

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

def generate_path_data(K = 3, max_feat = 8, max_path = 8, num_rel = 12, num_feat = 2850, num_visit = 6, num_target = 90):
    """
    generate path data including feat_index, path_index, path_target, and path_structure
    :param num_feat:          The number of the medical feature
    :param max_feat:          The maximum number of the patient feature in a visit
    :param num_rel:           The number of the relation types
    :param K:                 The length of the path
    :param num_path:          The maximum number of the paths linked with a feature
    :param max_path:          The maximum number of the paths linked with a feature
    :param num_visit:         The number of visit recorded in a sample
    :param num_target:        The number of label
    """
    with open('../data/mimic-iii/adjacent_list.pkl', 'rb') as f:
        adj_list = pickle.load(f)
    with open('../data/mimic-iii/adjacent_matrix.pkl', 'rb') as f:
        adj = pickle.load(f)
    features = torch.load('../data/mimic-iii/features_one_hot.pt')
    top = pd.read_csv('../data/mimic-iii/top_diagnoses.csv')
    top = top.sort_values('Diagnosis')
    icd = top.iloc[0:90]['Diagnosis']
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
                    if len(paths) > K:
                        return paths
        return paths

    total_path = 0
    paths = []
    for sample in tqdm(features, total = len(features), desc = 'generating paths'):
        sample_paths = []
        for visit in sample:
            visit_paths = {}
            for i in range(visit.shape[1]):
                if visit[0][i] != 0:
                    all_paths = []
                    all_paths = find_all_paths(i)
                    total_path += len(all_paths)
                    visit_paths[i] = all_paths
            sample_paths.append(visit_paths)
        sample_paths = sample_paths[::-1]   # Place paths in reverse chronological order
        paths.append(sample_paths)

    # feat_index: [sample_num * tensor(num_visit, max_feat, num_feat)]
    # path_index: [sample_num * tensor(num_visit, max_feat, max_path(path_id))] start from 1
    # path_target: [sample_num * tensor(num_path, num_target)]
    # path_structure: [sample_num * tensor(num_path, K, num_rel)]
    # num_path = num_visit * max_feat * max_path
    feat_index = []
    path_index = []
    path_target = []
    path_structure = []
    for sample in tqdm(paths, total = len(paths), desc = "generating relation"):
        sample_path_index = []
        sample_feat_index = []
        sample_path_target = torch.zeros(num_visit * max_feat * max_path, num_target)
        sample_path_structure = torch.zeros(num_visit * max_feat * max_path, K, num_rel)
        sample_path_count = 1
        for visit in sample:
            visit_path_index = torch.zeros(size=(max_feat, max_path))
            visit_feat_index = torch.zeros(size=(max_feat, num_feat))
            visit_feat = []
            for k, v in visit.items():
                ok = k
                feat_path_count = 0
                if k not in visit_feat:
                    if len(visit_feat) >= max_feat:
                        break
                    visit_feat.append(k)
                k = visit_feat.index(k)
                visit_feat_index[k][ok] = 1
                for path_idx, path in enumerate(v):
                    if feat_path_count < max_path:
                        target = path[-1]
                        target_index = target_idx.index(target)
                        sample_path_target[sample_path_count][target_index] = 1

                        visit_path_index[k][feat_path_count] = sample_path_count

                        for i in range(len(path) - 1):
                            sample_path_structure[sample_path_count][i][int(adj[path[i]][path[i+1]])] = 1
                        if len(path) - 1 < K:
                            for j in range(K - len(path) + 1):
                                sample_path_structure[sample_path_count][len(path) - 1 + j][0] = 1
                    else:
                        break

                    sample_path_count += 1
                    feat_path_count += 1
            sample_feat_index.append(visit_feat_index)
            sample_path_index.append(visit_path_index)
        sample_feat_index = torch.stack(sample_feat_index, dim=0)
        sample_path_index = torch.stack(sample_path_index, dim=0)
        feat_index.append(sample_feat_index)
        path_index.append(sample_path_index)
        path_target.append(sample_path_target)
        path_structure.append(sample_path_structure)
    try:
        torch.save(feat_index, f'../data/mimic-iii/dp_feat_index_{K}.pt')
        torch.save(path_index, f'../data/mimic-iii/dp_path_index_{K}.pt')
        torch.save(path_target, f'../data/mimic-iii/dp_path_target_{K}.pt')
        torch.save(path_structure, f'../data/mimic-iii/dp_path_structure_{K}.pt')
    except Exception as err:
        print(f'fail to save file:{err}')

def generate_medicationrecommendation_data():
    """generate medication recommendation data"""
    
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
    for sample in tqdm(features, total=len(features), desc="generating medicationrecommendation data"):
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
    torch.save(drugrecommendation_features_one_hot, '../data/mimic-iii/medicationrecommendation_features_one_hot.pt')
    torch.save(drugrecommendation_label_one_hot, '../data/mimic-iii/medicationrecommendation_label_one_hot.pt')

#----------------------------------------------------------------------generate_readmission_data-----------------------------------------------------------------------------------------------
<<<<<<< HEAD
def generate_readmission_path_data(K = 3, max_feat = 8, max_path = 8, num_rel = 12, num_feat = 2850, num_visit = 6, num_target = 90):
    """
    generate readmission prediction path data including feat_index, path_index, path_target, and path_structure
    :param num_feat:          The number of the medical feature
    :param max_feat:          The maximum number of the patient feature in a visit
    :param num_rel:           The number of the relation types
    :param K:                 The length of the path
    :param num_path:          The maximum number of the paths linked with a feature
    :param max_path:          The maximum number of the paths linked with a feature
    :param num_visit:         The number of visit recorded in a sample
    :param num_target:        The number of label
    """
    with open('adjacent_list.pkl', 'rb') as f:
=======

def generate_readmission_paths():
    """generate_readmission_paths"""
    K = 3 #path's length
    features = torch.load('readmission_features_one_hot.pt')
    with open('../data/mimic-iii/adjacent_list.pkl', 'rb') as f:
>>>>>>> 196112e0016ff747ae4250cfee641d8ea2a43547
        adj_list = pickle.load(f)
    with open('adjacent_matrix.pkl', 'rb') as f:
        adj = pickle.load(f)
    features = torch.load('features_one_hot.pt')
    top = pd.read_csv('top_diagnoses.csv')
    top = top.sort_values('Diagnosis')
    icd = top.iloc[0:90]['Diagnosis']
    target_idx = [code_map[i] for i in icd]

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

    total_path = 0
    paths = []
    for sample in tqdm(features, total = len(features), desc = 'generating paths'):
        sample_paths = []
        for visit in sample:
            # target_idx for readmission
            target_idx = [i for i in range(visit.shape[1]) if visit[0][i] != 0]
            visit_paths = {}
            for i in range(visit.shape[1]):
                if visit[0][i] != 0:
                    all_paths = []
                    all_paths = find_all_paths(i, target_idx)
                    total_path += len(all_paths)
                    visit_paths[i] = all_paths
            sample_paths.append(visit_paths)
        sample_paths = sample_paths[::-1]   # Place paths in reverse chronological order
        paths.append(sample_paths)
<<<<<<< HEAD

    # feat_index: [sample_num * tensor(num_visit, max_feat, num_feat)]
    # path_index: [sample_num * tensor(num_visit, max_feat, max_path(path_id))] start from 1
    # path_target: [sample_num * tensor(num_path, num_target)]
    # path_structure: [sample_num * tensor(num_path, K, num_rel)]
    # num_path = num_visit * max_feat * max_path
=======
    with open(f'../data/mimic-iii/readmission_paths_{K}.pkl', 'wb') as e:
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
    with open(f'../data/mimic-iii/readmission_paths_{K}.pkl', 'rb') as f:
        paths3 = pickle.load(f)
    with open('../data/mimic-iii/adjacent_matrix.pkl', 'rb') as f:
        adj = pickle.load(f)
    rel_index = []
>>>>>>> 196112e0016ff747ae4250cfee641d8ea2a43547
    feat_index = []
    path_index = []
    path_target = []
    path_structure = []
    for sample in tqdm(paths, total = len(paths), desc = "generating relation"):
        sample_path_index = []
        sample_feat_index = []
        sample_path_target = torch.zeros(num_visit * max_feat * max_path, num_target)
        sample_path_structure = torch.zeros(num_visit * max_feat * max_path, K, num_rel)
        sample_path_count = 1
        for visit in sample:
            visit_path_index = torch.zeros(size=(max_feat, max_path))
            visit_feat_index = torch.zeros(size=(max_feat, num_feat))
            visit_feat = []
            for k, v in visit.items():
                ok = k
                feat_path_count = 0
                if k not in visit_feat:
                    if len(visit_feat) >= max_feat:
                        break
                    visit_feat.append(k)
                k = visit_feat.index(k)
                visit_feat_index[k][ok] = 1
                for path_idx, path in enumerate(v):
                    if feat_path_count < max_path:
                        target = path[-1]
                        target_index = target_idx.index(target)
                        sample_path_target[sample_path_count][target_index] = 1

                        visit_path_index[k][feat_path_count] = sample_path_count

                        for i in range(len(path) - 1):
                            sample_path_structure[sample_path_count][i][int(adj[path[i]][path[i+1]])] = 1
                        if len(path) - 1 < K:
                            for j in range(K - len(path) + 1):
                                sample_path_structure[sample_path_count][len(path) - 1 + j][0] = 1
                    else:
                        break
<<<<<<< HEAD

                    sample_path_count += 1
                    feat_path_count += 1
            sample_feat_index.append(visit_feat_index)
            sample_path_index.append(visit_path_index)
        sample_feat_index = torch.stack(sample_feat_index, dim=0)
        sample_path_index = torch.stack(sample_path_index, dim=0)
        feat_index.append(sample_feat_index)
        path_index.append(sample_path_index)
        path_target.append(sample_path_target)
        path_structure.append(sample_path_structure)
    try:
        torch.save(feat_index, f'./rp_feat_index_{K}.pt')
        torch.save(path_index, f'./rp_path_index_{K}.pt')
        torch.save(path_target, f'./rp_path_target_{K}.pt')
        torch.save(path_structure, f'./rp_path_structure_{K}.pt')
    except Exception as err:
        print(f'fail to save file:{err}')
=======
            sample_feat.append(real_feat)
            sample_rel.append(visit_rel)
        sample_feat = torch.stack(sample_feat, dim=0)
        feat_index.append(sample_feat)
        sample_rel = torch.stack(sample_rel, dim=0)
        rel_index.append(sample_rel)
    print(len(rel_index))
    print(len(feat_index))
    torch.save(rel_index, f'../data/mimic-iii/readmission_rel_index_{K}.pt')
    torch.save(feat_index, f'../data/mimic-iii/readmission_feat_index_{K}.pt')
>>>>>>> 196112e0016ff747ae4250cfee641d8ea2a43547

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preprocessing')
    parser.add_argument('--task', type=str, default="diagnosis prediction", help='task description')
    args = parser.parse_args()

    generate_one_hot()
    if args.task == "diagnosis_prediction":
        generate_adjacent_list()
        generate_path_data()
    elif args.task == "medication_recommendation":
        generate_medicationrecommendation_data()
    elif args.task == "readmission_prediction":
        generate_readmission_path_data()
