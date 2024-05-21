import pickle
import numpy as np
import pandas as pd

import torch as th

import dgl


def load_data(ROOT_PATH, feature_length):
    # read data as a numpy arrays
    drug_drug = np.loadtxt(ROOT_PATH + 'mat_drug_drug.txt')
    drug_disease = np.loadtxt(ROOT_PATH + 'mat_drug_disease.txt')
    drug_protein = np.loadtxt(ROOT_PATH + 'mat_drug_protein.txt')
    drug_se = np.loadtxt(ROOT_PATH + 'mat_drug_se.txt')
    drug_similarity = np.loadtxt(ROOT_PATH + 'Similarity_Matrix_Drugs.txt')

    disease_drug = drug_disease.T
    se_drug = drug_se.T

    protein_disease = np.loadtxt(ROOT_PATH + 'mat_protein_disease.txt')
    protein_drug = np.loadtxt(ROOT_PATH + 'mat_protein_drug.txt')
    protein_protein = np.loadtxt(ROOT_PATH + 'mat_protein_protein.txt')
    protein_similarity = np.loadtxt(
        ROOT_PATH + 'Similarity_Matrix_Proteins.txt')
    disease_protein = protein_disease.T

    num_nodes = {'drug': drug_drug.shape[0], 'diease': drug_disease.shape[1], 
                        'sideeffect': drug_se.shape[1], 'protein': protein_protein.shape[0]}

    drug_drug = pd.DataFrame(drug_drug)
    drug_disease = pd.DataFrame(drug_disease)
    drug_protein = pd.DataFrame(drug_protein)
    drug_se = pd.DataFrame(drug_se)
    drug_similarity = pd.DataFrame(drug_similarity)
    disease_drug = pd.DataFrame(disease_drug)
    se_drug = pd.DataFrame(se_drug)

    protein_disease = pd.DataFrame(protein_disease)
    protein_drug = pd.DataFrame(protein_drug)
    protein_protein = pd.DataFrame(protein_protein)
    protein_similarity = pd.DataFrame(protein_similarity)
    protein_similarity = protein_similarity/100
    disease_protein = pd.DataFrame(disease_protein)

    drug_drug = pd.DataFrame(
        np.array(np.where(drug_drug == 1)).T, columns=['Drug1', 'Drug2'])
    drug_disease = pd.DataFrame(
        np.array(np.where(drug_disease == 1)).T, columns=['Drug', 'Disease'])
    drug_se = pd.DataFrame(
        np.array(np.where(drug_se == 1)).T, columns=['Drug', 'Se'])
    drug_similarity = pd.DataFrame(
        np.array(np.where(drug_similarity == 1)).T, columns=['Drug1', 'Drug2'])
    drug_protein = pd.DataFrame(
        np.array(np.where(drug_protein == 1)).T, columns=['Drug', 'Protein'])
    disease_drug = pd.DataFrame(
        np.array(np.where(disease_drug == 1)).T, columns=['Disease', 'Drug'])
    se_drug = pd.DataFrame(
        np.array(np.where(se_drug == 1)).T, columns=['Se', 'Drug'])

    protein_disease = pd.DataFrame(
        np.array(np.where(protein_disease == 1)).T, columns=['Protein', 'Disease'])
    protein_protein = pd.DataFrame(
        np.array(np.where(protein_protein == 1)).T, columns=['Protein1', 'Protein2'])
    protein_similarity = pd.DataFrame(np.array(
        np.where(protein_similarity == 1)).T, columns=['Protein1', 'Protein2'])
    disease_protein = pd.DataFrame(
        np.array(np.where(disease_protein == 1)).T, columns=['Disease', 'Protein'])
    protein_drug = pd.DataFrame(
        np.array(np.where(protein_drug == 1)).T, columns=['Protein', 'Drug'])

    g_data = {
        ('drug', 'dsim', 'drug'): (th.tensor(drug_similarity['Drug1'].values),
                                   th.tensor(drug_similarity['Drug2'].values)),
        ('drug', 'chemical', 'drug'): (th.tensor(drug_drug['Drug1'].values),
                                       th.tensor(drug_drug['Drug2'].values)),
        ('drug', 'ddi', 'diease'): (th.tensor(drug_disease['Drug'].values),
                                    th.tensor(drug_disease['Disease'].values)),
        ('diease', 'did', 'drug'): (th.tensor(disease_drug['Disease'].values),
                                    th.tensor(disease_drug['Drug'].values)),
        ('drug', 'dse', 'sideeffect'): (th.tensor(drug_se['Drug'].values),
                                        th.tensor(drug_se['Se'].values)),
        ('sideeffect', 'sed', 'drug'): (th.tensor(se_drug['Se'].values),
                                        th.tensor(se_drug['Drug'].values)),
        ('drug', 'ddp', 'protein'): (th.tensor(drug_protein['Drug'].values),
                                     th.tensor(drug_protein['Protein'].values)),
        ('protein', 'pdd', 'drug'): (th.tensor(protein_drug['Protein'].values),
                                     th.tensor(protein_drug['Drug'].values)),
        ('protein', 'psim', 'protein'): (th.tensor(protein_similarity['Protein1'].values),
                                         th.tensor(protein_similarity['Protein2'].values)),
        ('protein', 'sequence', 'protein'): (th.tensor(protein_protein['Protein1'].values),
                                             th.tensor(protein_protein['Protein2'].values)),
        ('protein', 'pdi', 'diease'): (th.tensor(protein_disease['Protein'].values),
                                       th.tensor(protein_disease['Disease'].values)),
        ('diease', 'dip', 'protein'): (th.tensor(disease_protein['Disease'].values),
                                       th.tensor(disease_protein['Protein'].values)),
    }

    g = dgl.heterograph(g_data,num_nodes_dict=num_nodes)
    d_len = len(g.nodes('drug'))
    p_len = len(g.nodes('protein'))
    se_len = len(g.nodes('sideeffect'))
    di_len = len(g.nodes('diease'))
    dim = d_len + p_len + di_len + se_len
    feature_d = np.random.randn(d_len, feature_length)
    feature_p = np.random.randn(p_len, feature_length)
    feature_di = np.random.randn(di_len, feature_length)
    feature_se = np.random.randn(se_len, feature_length)
    g.nodes['drug'].data['h'] = th.from_numpy(feature_d).to(th.float32)
    g.nodes['protein'].data['h'] = th.from_numpy(feature_p).to(th.float32)
    g.nodes['diease'].data['h'] = th.from_numpy(feature_di).to(th.float32)
    g.nodes['sideeffect'].data['h'] = th.from_numpy(feature_se).to(th.float32)

    

    type_mask = np.zeros((dim), dtype=int)
    type_mask[di_len:di_len+d_len] = 1
    type_mask[di_len+d_len:di_len+d_len+p_len] = 2
    type_mask[di_len+d_len+p_len:] = 3

    return g, type_mask


def load_meta_path(PATH):
    res = {}
    # Load Drugs metapathes
    in_file = open(PATH + 'd_p_d.idx', 'rb')
    res['drug_protein_drug'] = pickle.load(in_file)
    in_file.close()

    in_file = open(PATH + 'd_d.idx', 'rb')
    res['drug_drug'] = pickle.load(in_file)
    in_file.close()

    in_file = open(PATH + 'd_d_sim.idx', 'rb')
    res['drug_sim'] = pickle.load(in_file)
    in_file.close()

    in_file = open(PATH + 'd_se_d.idx', 'rb')
    res['drug_se_drug'] = pickle.load(in_file)
    in_file.close()

    in_file = open(PATH + 'd_di_d.idx', 'rb')
    res['drug_disease_drug'] = pickle.load(in_file)
    in_file.close()

    # Load protein metapathes
    in_file = open(PATH + 'p_di_p.idx', 'rb')
    res['protein_disease_protein'] = pickle.load(in_file)
    in_file.close()

    in_file = open(PATH + 'p_p.idx', 'rb')
    res['protein_protein'] = pickle.load(in_file)
    in_file.close()

    in_file = open(PATH + 'p_p_sim.idx', 'rb')
    res['protein_sim'] = pickle.load(in_file)
    in_file.close()

    in_file = open(PATH + 'p_dr_p.idx', 'rb')
    res['protein_drug_protein'] = pickle.load(in_file)
    in_file.close()
    return res


def load_meta_path_adj(PATH):
    res = {}
    file = open(PATH + 'd_p_d.adjlist', 'r')
    res['drug_protein_drug'] = [line.strip() for line in file]
    file = open(PATH + 'd_se_d.adjlist', 'r')
    res['drug_se_drug'] = [line.strip() for line in file]
    file = open(PATH + 'd_d.adjlist', 'r')
    res['drug_drug'] = [line.strip() for line in file]
    file = open(PATH + 'd_d_sim.adjlist', 'r')
    res['drug_sim'] = [line.strip() for line in file]
    file = open(PATH + 'd_di_d.adjlist', 'r')
    res['drug_disease_drug'] = [line.strip() for line in file]
    file = open(PATH + 'p_di_p.adjlist', 'r')
    res['protein_disease_protein'] = [line.strip() for line in file]
    file = open(PATH + 'p_p.adjlist', 'r')
    res['protein_protein'] = [line.strip() for line in file]
    file = open(PATH + 'p_p_sim.adjlist', 'r')
    res['protein_sim'] = [line.strip() for line in file]
    file = open(PATH + 'p_dr_p.adjlist', 'r')
    res['protein_drug_protein'] = [line.strip() for line in file]
    return res


def generate_pos_neg_by_edge(ROOT_PATH, num_drugs, num_proteins, type=0):

    drug_protein = np.loadtxt(ROOT_PATH + 'mat_drug_protein.txt')
    all_positive = []
    whole_neg = []
    pos_count = 0
    neg_count = 0
    for i in range(num_drugs):
        for j in range(num_proteins):
            if int(drug_protein[i][j]) == 1:
                all_positive.append([i, j])
                pos_count += 1
            elif int(drug_protein[i][j]) == 0:
                whole_neg.append([i, j])
                neg_count += 1

    if type == 0:
        neg_len = len(all_positive)
        neg_index = np.random.choice(
            np.arange(len(whole_neg)), size=neg_len, replace=False)
    elif type == 1:
        neg_len = len(all_positive) * 3
        neg_index = np.random.choice(
            np.arange(len(whole_neg)), size=neg_len, replace=False)
    elif type == 2:
        neg_len = len(all_positive) * 5
        neg_index = np.random.choice(
            np.arange(len(whole_neg)), size=neg_len, replace=False)
    elif type == 3:
        neg_len = len(all_positive) * 10
        neg_index = np.random.choice(
            np.arange(len(whole_neg)), size=neg_len, replace=False)
    elif type == 4:
        neg_index = np.random.choice(
            np.arange(len(whole_neg)), size=len(whole_neg), replace=False)

    all_positive = np.array(
        all_positive, dtype=int).reshape((pos_count, 2))
    all_negative = []
    for i in neg_index:
        all_negative.append(whole_neg[i])
    all_negative = np.array(all_negative)
    np.random.shuffle(all_positive)
    np.random.shuffle(all_negative)
    

    data_set = np.zeros((len(all_negative)+len(all_positive), 3), dtype=int)
    #print (data_set)
    count = 0
    for i in all_positive:
        data_set[count][0] = i[0]
        data_set[count][1] = i[1]
        data_set[count][2] = 1
        count += 1
    for i in all_negative:
        data_set[count][0] = i[0]
        data_set[count][1] = i[1]
        data_set[count][2] = 0
        count += 1

    return data_set


def getData(feature_length, mode=0):
    '''
    * mode 0 means the sampling is one-one
    * mode 1 means the sampling is one+ ten-
    * mode 2 means the sampling is one+ to all-
    # '''

    # Step 1. Load Graph
    GRAPH_DATA_PATH = './data/DTINet/'
    adjMat, type_mask = load_data(
        GRAPH_DATA_PATH, feature_length)

    # Step 2. Load Meta pathes indices
    METAPATH_DATA_PATH = './data/metapathes/'
    metaPath_dict = load_meta_path(METAPATH_DATA_PATH)

    # Step 3: Load Meta pathes adjList
    METAPATH_DATA_PATH = './data/metapathes/'
    metaPath_adj_dict = load_meta_path_adj(METAPATH_DATA_PATH)

    # Step 4: Generate the positive and negative based on mode
    num_drug = len(adjMat.nodes('drug'))
    num_protein = len(adjMat.nodes('protein'))

    data_set = generate_pos_neg_by_edge(
        GRAPH_DATA_PATH, num_drug, num_protein, mode)
    all_mp = {
        'drug': {
            'drug_sim': ['dsim'],
            'drug_drug': ['chemical'],
            'drug_disease_drug': ['ddi', 'did'],
            'drug_se_drug': ['dse', 'sed'],
            'drug_protein_drug': ['ddp', 'pdd']},
        'protein': {
            'protein_sim': ['psim'],
            'protein_protein': ['sequence'],
            'protein_disease_protein': ['pdi', 'dip'],
            'protein_drug_protein': ['pdd', 'ddp']}
    }
    # mask_key_name = ['drug_protein_drug', 'protein_drug_protein']

    return metaPath_adj_dict, metaPath_dict, adjMat, type_mask, data_set, all_mp
