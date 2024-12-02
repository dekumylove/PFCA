from tqdm import tqdm
import torch
import pickle
import requests



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

    adj_m = torch.zeros(size = (len(text_cui), len(text_cui)))  #init adjacent matrix
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
    extract_relations()
    extract_triplets()
    generate_adjacent_matrix()