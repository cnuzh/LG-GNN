import numpy as np
import torch
from utils.gcn_utils import preprocess_features
from sklearn.model_selection import StratifiedKFold
from node import get_node_feature
import csv
import os
import argparse
from networkx.convert_matrix import from_numpy_matrix
import networkx as nx
from torch_geometric.utils import remove_self_loops
from torch_sparse import coalesce
import scipy.io as sio
from torch_geometric.data import InMemoryDataset,Data
from scipy.spatial import distance

data_folder = "/...."

def standardization_intensity_normalization(dataset, dtype):
    mean = dataset.mean()
    std  = dataset.std()
    return ((dataset - mean) / std).astype(dtype)
def intensityNormalisationFeatureScaling(dataset, dtype):
    max = dataset.max()
    min = dataset.min()

    return ((dataset - min) / (max - min)).astype(dtype)
class dataloader():
    def __init__(self): 
        self.pd_dict = {}
        self.node_ftr_dim = 2000  ##2000
        self.num_classes = 2 

    def load_data(self, connectivity='correlation', atlas='ho'):

        subject_IDs = get_ids()
        labels = get_subject_score(subject_IDs, score='Group')
        num_nodes = len(subject_IDs)
        ages = get_subject_score(subject_IDs, score='Age')
        genders = get_subject_score(subject_IDs, score='Gender') 
        y_onehot = np.zeros([num_nodes, self.num_classes])
        y = np.zeros([num_nodes])
        age = np.zeros([num_nodes], dtype=np.float32)
        gender = np.zeros([num_nodes], dtype=np.int)
        for i in range(num_nodes):
            y_onehot[i, int(labels[subject_IDs[i]])-1] = 1
            y[i] = int(labels[subject_IDs[i]])
            age[i] = float(ages[subject_IDs[i]])
            gender[i] = genders[subject_IDs[i]]
        self.y = y 
        self.raw_features = get_node_feature()
        phonetic_data = np.zeros([num_nodes, 2], dtype=np.float32)
        phonetic_data[:,0] = gender 
        phonetic_data[:,1] = age 
        self.pd_dict['Gender'] = np.copy(phonetic_data[:,0])
        self.pd_dict['Age'] = np.copy(phonetic_data[:,1])
        phonetic_score = self.pd_dict
        return self.raw_features, self.y, phonetic_data, phonetic_score 
    

    def data_split(self, n_folds):
        skf = StratifiedKFold(n_splits=n_folds)
        cv_splits = list(skf.split(self.raw_features, self.y))
        return cv_splits 



    def get_PAE_inputs(self, nonimg):
 
        n = self.node_ftr.shape[0] 
        num_edge = n*(1+n)//2 - n  
        pd_ftr_dim = nonimg.shape[1]
        edge_index = np.zeros([2, num_edge], dtype=np.int64) 
        edgenet_input = np.zeros([num_edge, 2*pd_ftr_dim], dtype=np.float32)  
        aff_score = np.zeros(num_edge, dtype=np.float32)   
        aff_adj = get_static_affinity_adj(self.node_ftr, self.pd_dict)  
        flatten_ind = 0 
        for i in range(n):
            for j in range(i+1, n):
                edge_index[:,flatten_ind] = [i,j]
                edgenet_input[flatten_ind]  = np.concatenate((nonimg[i], nonimg[j]))
                aff_score[flatten_ind] = aff_adj[i][j]  
                flatten_ind +=1

        assert flatten_ind == num_edge, "Error in computing edge input"
        
        keep_ind = np.where(aff_score > 1.1)[0]  
        edge_index = edge_index[:, keep_ind]
        edgenet_input = edgenet_input[keep_ind]

        return edge_index, edgenet_input
    def get_inputs(self, nonimg, embeddings, phonetic_score):
 
        
        self.node_ftr  = np.array(embeddings.detach().cpu().numpy())
        n = self.node_ftr.shape[0] 
        num_edge = n*(1+n)//2 - n  
        pd_ftr_dim = nonimg.shape[1]
        edge_index = np.zeros([2, num_edge], dtype=np.int64) 
        edgenet_input = np.zeros([num_edge, 2*pd_ftr_dim], dtype=np.float32)  
        aff_score = np.zeros(num_edge, dtype=np.float32)   
        aff_adj = get_static_affinity_adj(self.node_ftr, phonetic_score)  
        flatten_ind = 0 
        for i in range(n):
            for j in range(i+1, n):
                edge_index[:,flatten_ind] = [i,j]
                edgenet_input[flatten_ind]  = np.concatenate((nonimg[i], nonimg[j]))
                aff_score[flatten_ind] = aff_adj[i][j]  
                flatten_ind +=1

        assert flatten_ind == num_edge, "Error in computing edge input"
        
        keep_ind = np.where(aff_score > 1.1)[0]  
        edge_index = edge_index[:, keep_ind]
        edgenet_input = edgenet_input[keep_ind]

        return edge_index, edgenet_input
def get_subject_score(subject_list, score):
    scores_dict = {}

    phenotype = "/phenotypic_information.csv" 
    with open(phenotype) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['Image Data ID'][1:] in subject_list:  
                scores_dict[row['Image Data ID'][1:]] = row[score]
    return scores_dict
def get_ids(num_subjects=None):

    subject_IDs = np.genfromtxt(os.path.join("timeseries_subjects_id.txt"), dtype=str)
    if num_subjects is not None:
        subject_IDs = subject_IDs[:num_subjects]
    return subject_IDs




def create_affinity_graph_from_scores(scores, pd_dict):
    num_nodes = len(pd_dict[scores[0]]) 
    graph = np.zeros((num_nodes, num_nodes))

    for l in scores:
        label_dict = pd_dict[l]

        if l in ['AGE_AT_SCAN', 'FIQ']:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    try:
                        val = abs(float(label_dict[k]) - float(label_dict[j]))
                        if val < 2:
                            graph[k, j] += 1
                            graph[j, k] += 1
                    except ValueError:  # missing label
                        pass

        else:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    if label_dict[k] == label_dict[j]:
                        graph[k, j] += 1
                        graph[j, k] += 1

    return graph

def get_static_affinity_adj(features, pd_dict):
    pd_affinity = create_affinity_graph_from_scores(['SITE_ID','SEX', 'AGE_AT_SCAN'], pd_dict)
    distv = distance.pdist(features, metric='correlation') 
    dist = distance.squareform(distv)  
    sigma = np.mean(dist)
    feature_sim = np.exp(- dist ** 2 / (2 * sigma ** 2))
    adj = pd_affinity * feature_sim  

    return adj
