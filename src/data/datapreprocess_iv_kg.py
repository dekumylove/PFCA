import torch
import pickle
import requests
import subprocess
import re
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter
from tqdm import tqdm
# select the 90 most common diagnoses
def select_top_diagnoses():
    diagnoses = pd.read_csv('../data/mimic-iv/diagnoses_icd.csv')
    dia_counter = Counter(diagnoses.iloc[:, 3].astype(str))
    top_diagnoses = dia_counter.most_common(90)
    df_top_diagnoses = pd.DataFrame(top_diagnoses, columns=['Diagnosis', 'Count'])
    df_top_diagnoses.to_csv('../data/mimic-iv/top_diagnoses_iv.csv', index=False)

# filter samples based on the 90 most common diagnoses
def filter_visits(ratio = 0.7):
    filtered_visit = []
    patient_diagnoses = pd.read_csv('../data/mimic-iv/diagnoses_icd.csv')
    top_diagnoses = pd.read_csv('../data/mimic-iv/top_diagnoses_iv.csv')

    top_diagnoses_list = list(top_diagnoses.loc[:,'Diagnosis'])
    group_list = ['subject_id', 'hadm_id']
    patient_diagnoses = patient_diagnoses.groupby(group_list)
    for key, group_df in tqdm(patient_diagnoses, total = len(patient_diagnoses), desc = "filtered_patients"):
        sum = 0
        dia_codes = group_df['icd_code']
        for code in dia_codes:
            if code in top_diagnoses_list:
                sum += 1
        if sum >= len(dia_codes) * ratio:
            filtered_visit.append(list(key))
        
    filtered_visits_df = pd.DataFrame(filtered_visit, columns = ['subject_id', 'hadm_id'])
    filtered_visits_df_no_dup = filtered_visits_df.drop_duplicates()
    print(len(filtered_visits_df), len(filtered_visits_df_no_dup))
    filtered_visits_df.to_csv(f'data/mimic-iv/filtered_visits_{ratio}_iv.csv')

# generate samples according to the visits filtered
def generate_sample():
    diagnose_data = pd.read_csv('../data/mimic-iv/admissions.csv')
    filtered_visits = pd.read_csv('../data/mimic-iv/filtered_visits_0.7_iv.csv')
    diagnose_data_grouped = diagnose_data.groupby('subject_id')
    diagnose_data_nodup = diagnose_data.drop_duplicates(subset=['subject_id'])
    print(len(diagnose_data), len(diagnose_data_nodup))
    # select all visits from ADMISSIONS within 120 days 
    labels = []
    features = []
    predate = datetime.min
    only_one_num = 0
    sample_num = 0
    for index, filtered in tqdm(filtered_visits.iterrows(), total = len(filtered_visits), desc = "filtered_visits"):
        patients = diagnose_data_grouped.get_group(filtered['subject_id'])
        patients = patients.sort_values(by='admittime', ascending=False).reset_index()
        root_index = patients.loc[patients['hadm_id'] == filtered['hadm_id']].index     #find the index of this visit
        date_0 = datetime.strptime(patients.iloc[root_index.item()]['admittime'], '%Y-%m-%d %H:%M:%S')
        predate = date_0
        counts = 0
        sub_labels = []
        sub_features = []
        window_num = 1
        
        for i in range(root_index.item(), len(patients)):
            date_1 = datetime.strptime(patients.iloc[i]['admittime'], '%Y-%m-%d %H:%M:%S')
            time_difference_1 = date_0 - date_1  #120days' window
            
            if time_difference_1.days > 120:
                break
            
            time_difference_2 = predate - date_1 #20days' window
            if time_difference_2.days <= 20:
                counts += 1
                sub_labels.append(f"{filtered['subject_id']}_{filtered['hadm_id']}")
                sub_features.append(f"{filtered['subject_id']}_{patients.iloc[i]['hadm_id']}_{window_num}")
            else:
                window_num += 1
                predate = predate - timedelta(days=20)
                time_difference_2 = predate - date_1
                while(time_difference_2.days > 20):
                    window_num += 1
                    predate = predate - timedelta(days=20)
                    time_difference_2 = predate - date_1
                sub_labels.append(f"{filtered['subject_id']}_{filtered['hadm_id']}")
                sub_features.append(f"{filtered['subject_id']}_{patients.iloc[i]['hadm_id']}_{window_num}")
                counts += 1
                
        if counts == 1:
            only_one_num += 1
        else:   #find an available sample
            labels += sub_labels
            features += sub_features
            sample_num += 1

        window_num = 1
        sub_features = []
        sub_labels = []
    sample_df = pd.DataFrame({'labels':labels, 'features':features})
    sample_df.to_csv('../data/mimic-iv/sample_0.7_iv.csv', index = False)
    print("only_one_num: ", only_one_num)
    print('sample_num:',sample_num)

# generate code_map based on the sample
def get_code_map():
    diagnose_data = pd.read_csv('../data/mimic-iv/diagnoses_icd.csv')
    procedure_data = pd.read_csv('../data/mimic-iv/procedures_icd.csv')
    prescription_data = pd.read_csv('../data/mimic-iv/prescriptions.csv')
    samples = pd.read_csv('../data/mimic-iv/sample_0.7_iv.csv')
    print(len(samples))
    samples_list = [row['features'].split('_') for _,row in samples.iterrows()]
    samples_done = [f"{sample[0]}_{sample[1]}" for sample in samples_list]


    dia_code_map = {}
    pro_code_map = {}
    drug_code_map = {}
    dia_code_count = {}
    pro_code_count = {}
    drug_code_count = {}
    pos = 0
    for _,row in tqdm(diagnose_data.iterrows(), total = len(diagnose_data), desc = "diagnose_data"):
        key = f"{row['subject_id']}_{row['hadm_id']}"
        if key in samples_done:
            icd_code = row['icd_code']
            icd_version = row['icd_version']
            key = f'{icd_code}_{icd_version}'
            if key not in dia_code_map:
                dia_code_map[key] = pos
                dia_code_count[key] = 1
                pos += 1
            else:
                dia_code_count[key] += 1
    for key, value in dia_code_count.items():
        if value < 10:
            dia_code_map.pop(key)
    print(len(dia_code_map))

    for _,row in tqdm(procedure_data.iterrows(), total = len(procedure_data), desc = "procedure_data"):
        key = f"{row['subject_id']}_{row['hadm_id']}"
        if key in samples_done:
            icd_code = row['icd_code']
            icd_version = row['icd_version']
            key = f'{icd_code}_{icd_version}'
            if key not in pro_code_map:
                pro_code_map[key] = pos
                pro_code_count[key] = 1
                pos += 1
            else:
                pro_code_count[key] += 1
    for key, value in pro_code_count.items():
        if value < 10:
            pro_code_map.pop(key)    
    print(len(pro_code_map))

    for _,row in tqdm(prescription_data.iterrows(), total = len(prescription_data), desc = "prescription_data"):
        key = f"{row['subject_id']}_{row['hadm_id']}"
        if key in samples_done:
            drug = row['drug']
            icd_version = row['icd_version']
            key = f'{icd_code}_{icd_version}'
            if key not in drug_code_map:
                drug_code_map[row['drug']] = pos
                drug_code_count[row['drug']] = 1
                pos += 1
            else:
                drug_code_count[row['drug']] += 1
    for key, value in drug_code_count.items():
        if value < 10:
            drug_code_map.pop(key)
    print(len(drug_code_map))

    code_map = {}
    index = 0
    for k, v in dia_code_map.items():
        if k not in code_map:
            code_map[k] = index
            index += 1
    for k, v in pro_code_map.items():
        if k not in code_map:
            code_map[k] = index
            print(k, index)
            index += 1
    for k, v in drug_code_map.items():
        if k not in code_map:
            code_map[k] = index
            index += 1
    print(len(code_map))
    with open('../data/mimic-iv/code_map_10.pkl', 'wb') as f:
        pickle.dump(code_map, f)

# convert icd_code to icd_text
def icd_to_text():
    dia_to_text = {}
    pro_to_text = {}
    drug_to_text = {}
    with open('../data/mimic-iv/dia_code_map_10.pkl', 'rb') as f:
        dia_code_map = pickle.load(f)
        for key, value in tqdm(dia_code_map.items(), total = len(dia_code_map), desc = "converting dia to text"):
            key1 = key.split('_')
            if int(key1[1]) == 10 and len(key1[0]) > 3:
                code = key1[0][:3] + '.' + key1[0][3:]
            else:
                code = key1[0]
            url = f'https://clinicaltables.nlm.nih.gov/api/icd9cm_dx/v3/search?terms={code}'
            response = requests.get(url = url)
            if response:
                result = eval(response.text.replace('null', '" "'))
                if len(result[3]) > 0:
                    dia_to_text[code] = result[3]
                else:   
                    url2 = f'https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search?terms={code}'
                    response2 = requests.get(url = url2)
                    if response2:
                        result2 = eval(response2.text.replace('null', '" "'))
                        if len(result2[3]) > 0:
                            dia_to_text[code] = result2[3]
                        else:
                            dia_to_text[code] = [[code,' ']]
                            print(f'dia_to_text is: {dia_to_text[code]}_{key1[1]}')
        
        print(f'len(dia_to_text) is: {len(dia_to_text)}')
        with open('../data/mimic-iv/dia_to_text.pkl', 'wb') as e:
            pickle.dump(dia_to_text, e)

    with open('../data/mimic-iv/pro_code_map_10.pkl', 'rb') as f:
        pro_code_map = pickle.load(f)
        for key, value in tqdm(pro_code_map.items(), total = len(pro_code_map), desc = "converting pro to text"):
            key1 = key.split('_')
            if int(key1[1]) == 10 and len(key1[0]) > 3:
                code = key1[0][:3] + '.' + key1[0][3:]
            else:
                code = key1[0]
            url = f'https://clinicaltables.nlm.nih.gov/api/icd9cm_sg/v3/search?terms={code}'
            response = requests.get(url = url)
            if response:
                result = eval(response.text.replace('null', '" "'))
                if len(result[3]) > 0:
                    pro_to_text[code] = result[3]
                else:  
                    url2 = f'https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search?terms={code}'
                    response2 = requests.get(url = url2)
                    if response2:
                        result2 = eval(response2.text.replace('null', '" "'))
                        if len(result2[3]) > 0:
                            pro_to_text[code] = result2[3]
                        else:
                            pro_to_text[code] = [[code,' ']]
                            print(f'pro_to_text is: {pro_to_text[code]}_{key1[1]}')
        
        print(f'len(pro_to_text) is: {len(pro_to_text)}')
        with open('../data/mimic-iv/pro_to_text.pkl', 'wb') as e:
            pickle.dump(pro_to_text, e)
    with open('../data/mimic-iv/pro_to_text.pkl', 'rb') as f:
        pro_to_text = pickle.load(f)
        with open('../data/mimic-iv/pro_to_text.pkl', 'wb') as e:
            pickle.dump(pro_to_text, e)

    with open('../data/mimic-iv/drug_code_map_10.pkl', 'rb') as f:
        drug_code_map = pickle.load(f)
        for key, value in tqdm(drug_code_map.items(), total = len(drug_code_map), desc = "converting drug to text"):
            drug_to_text[key] = [[key,key]]
        
        print(f'len(drug_to_text) is: {len(drug_to_text)}')
        with open('../data/mimic-iv/drug_to_text.pkl', 'wb') as e:
            pickle.dump(drug_to_text, e)

# convert icd_text to CUI
def text_to_cui():
    dia_to_cui = {}
    with open('../data/mimic-iv/dia_to_text.pkl', 'rb') as f:
        dia_to_text = pickle.load(f)
        print(len(dia_to_text))
        for key, values in tqdm(dia_to_text.items(), total = len(dia_to_text), desc = "dia_to_text"):
            query = "echo " + "\"" + values[0][1] + "\"" + " | ../public_mm/bin/metamap -I;"
            result = subprocess.check_output(query, shell=True, text=True)
            pattern = r'Processing\s+(.+?):\s+(.+)'
            match = re.search(pattern, result)
            pattern2 = r'\bC\d+\b'
            matches = re.findall(pattern2, result)
            
            if match and matches:
                input_text = match.group(2)
                cui_code = matches[0]
                print(f'input_text:{input_text}')
                print(f'cui_code:{cui_code}')
                if input_text not in dia_to_cui:
                    dia_to_cui[input_text] = cui_code
            else:
                if key not in dia_to_cui:
                    dia_to_cui[key] = 'C0000000'
                print(f"no matching result for {key}")
    print(len(dia_to_cui))
    with open('../data/mimic-iv/dia_cui.pkl', 'wb') as f:
        pickle.dump(dia_to_cui, f)

    pro_to_cui = {}
    with open('../data/mimic-iv/pro_to_text.pkl', 'rb') as f:
        pro_to_text = pickle.load(f)
        print(len(pro_to_text))
        for key, values in tqdm(pro_to_text.items(), total = len(pro_to_text), desc = "pro_to_text"):
            query = "echo " + "\"" + values[0][1] + "\"" + " | ../public_mm/bin/metamap -I;"
            result = subprocess.check_output(query, shell=True, text=True)
            pattern = r'Processing\s+(.+?):\s+(.+)'
            match = re.search(pattern, result)
            pattern2 = r'\bC\d+\b'
            matches = re.findall(pattern2, result)
            if matches and match:
                input_text = match.group(2)
                cui_code = matches[0]
                print(f'input_text:{input_text}')
                print(f'cui_code:{cui_code}')
                if input_text not in pro_to_cui:
                    pro_to_cui[input_text] = cui_code
                else:
                    input_text = f'{input_text}_2'
                    pro_to_cui[input_text] = cui_code
                    print(input_text, cui_code)
            else:
                if key not in pro_to_cui:
                    pro_to_cui[key] = 'C0000000'
                else:
                    input_text = f'{key}_2'
                    pro_to_cui[input_text] = 'C0000000'
                print(f"no matching result for {key}")
    print(len(pro_to_cui))
    with open('../data/mimic-iv/pro_cui.pkl', 'wb') as f:
        pickle.dump(pro_to_cui, f)

    drug_to_cui = {}
    with open('../data/mimic-iv/drug_to_text.pkl', 'rb') as f:
        drug_to_text = pickle.load(f)
        for key, values in tqdm(drug_to_text.items(), total = len(drug_to_text), desc = "drug_to_text"):
            query = "echo " + "\"" + values + "\"" + " | ../public_mm/bin/metamap -I;"
            result = subprocess.check_output(query, shell=True, text=True)
            pattern = r'Processing\s+(.+?):\s+(.+)'
            match = re.search(pattern, result)
            pattern2 = r'\bC\d+\b'
            matches = re.findall(pattern2, result)
            
            if match and matches:
                input_text = match.group(2)
                cui_code = matches[0]
                print(f'input_text:{input_text}')
                print(f'cui_code:{cui_code}')
                if input_text not in drug_to_cui:
                    drug_to_cui[input_text] = cui_code
                else:
                    input_text = f'{input_text}_2'
                    drug_to_cui[input_text] = cui_code
                    print(input_text, cui_code)
            else:
                if key not in drug_to_cui:
                    drug_to_cui[key] = 'C0000000'
                else:
                    input_text = f'{key}_2'
                    drug_to_cui[input_text] = 'C0000000'
                print(f"no matching result for {key}")
    with open('../data/mimic-iv/drug_cui.pkl', 'wb') as f:
        pickle.dump(drug_to_cui, f)


def extract_relations():
    """
    extract relations according to diagnosis, medication, or procedure cui
    """ 
    with open('../data/mimic-iv/text_cui.pkl', 'rb') as f:
        text_cui = pickle.load(f)
    text_cui = text_cui.values()

    relations_list = {}
    with open("../data/UMLS/MRREL.RRF", "r", encoding="utf-8") as f:
        for line in tqdm(f, desc = "extracting relations", unit = "lines"):
            line = line.strip().split('|')
            CUI_1 = line[0]
            CUI_2 = line[4]
            if CUI_1 in text_cui:
                if CUI_2 in text_cui:
                    key = f'{CUI_1},{CUI_2}'
                    if key not in relations_list:
                        print(f'{key}:{line[3]}')
                        relations_list[key] = line[3]

    with open('../data/mimic-iv/relations.pkl', 'wb') as f:
        pickle.dump(relations_list, f)

def extract_triplets():
    """
    extract triplets from relations
    """

    converted_list = []
    icd_map = {}

    with open('../data/mimic-iv/relations.pkl', 'rb') as f:
        relations = pickle.load(f)
        print(len(relations))

        # use api to convert medical text to corresponding icd code
        for t in tqdm(relations, total = len(relations), desc = "extracting triplets"):
            if t[0] not in icd_map and t[3] not in icd_map:
                url1 = 'https://clinicaltables.nlm.nih.gov/api/icd9cm_dx/v3/search?terms=' + t[0]
                url2 = 'https://clinicaltables.nlm.nih.gov/api/icd9cm_dx/v3/search?terms=' + t[3]
                response1 = requests.get(url=url1)
                response2 = requests.get(url=url2)
                if response1 and response2:
                    text1 = response1.text.replace('null', '" "')
                    text2 = response2.text.replace('null', '" "')
                    rlist1 = eval(text1)
                    rlist2 = eval(text2)
                    if len(rlist1[1]) > 0 and len(rlist2[1]) > 0:
                        converted_concept = [rlist1[1], t[1], t[2], rlist2[1]]
                        converted_list.append(converted_concept)
                    icd_map[t[0]] = rlist1[1]
                    icd_map[t[3]] = rlist2[1]
            elif t[0] not in icd_map:
                url1 = 'https://clinicaltables.nlm.nih.gov/api/icd9cm_dx/v3/search?terms=' + t[0]
                response1 = requests.get(url=url1)
                if response1:
                    text1 = response1.text.replace('null', '" "')
                    rlist1 = eval(text1)
                    if len(rlist1[1]) > 0 and len(icd_map[t[3]]) > 0:
                        converted_concept = [rlist1[1], t[1], t[2], icd_map[t[3]]]
                        converted_list.append(converted_concept)
                    icd_map[t[0]] = rlist1[1]
            elif t[3] not in icd_map:
                url2 = 'https://clinicaltables.nlm.nih.gov/api/icd9cm_dx/v3/search?terms=' + t[3]
                response2 = requests.get(url=url2)
                if response2:
                    text2 = response2.text.replace('null', '" "')
                    rlist2 = eval(text2)
                    if len(rlist2[1]) > 0 and len(icd_map[t[0]]) > 0:
                        converted_concept = [icd_map[t[0]], t[1], t[2], rlist2[1]]
                        converted_list.append(converted_concept)
                    icd_map[t[3]] = rlist2[1]
            else:
                if len(icd_map[t[3]]) > 0 and len(icd_map[t[0]]) > 0:
                    converted_concept = [icd_map[t[0]], t[1], t[2], icd_map[t[3]]]
                    converted_list.append(converted_concept)
        
        with open('../data/mimic-iv/icd_triples.pkl', 'wb') as f:
            pickle.dump(converted_list, f)

def generate_adjacent_matrix():
    """
    generate adjacent matrix of the knowledge graph
    """
    with open('../data/mimic-iv/text_cui.pkl', 'rb') as f:
        text_cui = pickle.load(f)

    with open('../data/mimic-iv/relations.pkl', 'rb') as f:
        triples = pickle.load(f)

    adj_m = torch.zeros(size = (len(text_cui), len(text_cui)))
    cui_values = list(text_cui.values())
    relation_type = {}
    edge_num = 0
    for k, v in tqdm(triples.items(), total = len(triples), desc = "generating adjacent matrix"):
        cui = k.split(',')
        if cui[0] in cui_values and cui[1] in cui_values:
            edge_num += 1
            if v not in relation_type:
                relation_type[v] = len(relation_type) + 1
            index_1 = cui_values.index(cui[0])
            index_2 = cui_values.index(cui[1])
            adj_m[index_1][index_2] = relation_type[v]

    with open('../data/mimic-iv/adjacent_matrix.pkl', 'wb') as f:
        pickle.dump(adj_m, f)

if __name__ == '__main__':
    select_top_diagnoses()
    filter_visits()
    generate_sample()
    get_code_map()
    icd_to_text()
    text_to_cui()
    extract_relations()
    extract_triplets()
    generate_adjacent_matrix()