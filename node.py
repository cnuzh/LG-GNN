import os
import numpy as np
import deepdish
import argparse
import scipy.io as sio
import torch
from torch_geometric.data import InMemoryDataset,Data
from os.path import join, isfile
from os import listdir
import numpy as np
import os.path as osp
from networkx.convert_matrix import from_numpy_matrix
import networkx as nx
from torch_geometric.utils import remove_self_loops
from torch_sparse import coalesce


data_folder = "/..../" 
def read_sigle_data(data):
    pcorr = np.abs(data)
    num_nodes = pcorr.shape[0]
    G = from_numpy_matrix(pcorr)
    A = nx.to_scipy_sparse_matrix(G)
    adj = A.tocoo()
    edge_att = np.zeros(len(adj.row))
    for i in range(len(adj.row)):
        edge_att[i] = pcorr[adj.row[i], adj.col[i]]
    edge_index = np.stack([adj.row, adj.col])
    edge_index, edge_att = remove_self_loops(torch.from_numpy(edge_index), torch.from_numpy(edge_att))
    edge_index = edge_index.long()
    edge_index, edge_att = coalesce(edge_index, edge_att, num_nodes,
                                    num_nodes)
    att = data
    att[att== float('inf')] = 0
    att_torch = torch.from_numpy(att).float()
    graph = Data(x=att_torch, edge_index=edge_index.long(), edge_attr=edge_att)
    
    return graph
def get_networks(subject_list, kind, atlas_name="aal", variable='connectivity'):

    graphs = []
    for subject in subject_list:
        fl = os.path.join(data_folder, subject,
                          subject + "_" + atlas_name + "_" + kind + ".mat")
        matrix = sio.loadmat(fl)[variable]
        with np.errstate(divide='ignore', invalid='ignore'):
            norm_matrix = np.arctanh(matrix)
        graph = read_sigle_data(norm_matrix)               
        graphs.append(graph)

    return graphs

def get_node_feature():
    
    parser = argparse.ArgumentParser(description='Classification of the ABIDE dataset')
    parser.add_argument('--atlas', default='ho')
    parser.add_argument('--seed', default=123, type=int, help='Seed for random initialisation. default: 1234.')
    parser.add_argument('--nclass', default=2, type=int, help='Number of classes. default:2')
    args = parser.parse_args()
    params = dict()
    params['seed'] = args.seed  
    params['atlas'] = args.atlas 
    atlas = args.atlas  
    subject_IDs = Reader.get_ids()
    num_classes = args.nclass
    num_subjects = len(subject_IDs)
    params['n_subjects'] = num_subjects
    raw_feature = get_networks(subject_IDs, kind='correlation', atlas_name=atlas)

    return raw_feature





