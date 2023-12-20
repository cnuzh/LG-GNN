import torch
from torch_geometric.nn import GCNConv,ChebConv
from util  import dataloader
from torch.nn import Linear as Lin
from edge import EDGE
from opt import *
import torch.nn.functional as F
from torch import nn 
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from torch_geometric.nn.pool.topk_pool import topk,filter_adj
from dgl import DGLGraph
from dgl.nn import GraphConv
import dgl
from ASAPooling import ASAP_Pooling
from CayleyNet import CayleyConv
opt = OptInit().initialize()




class Local_GNN(torch.nn.Module):

    def __init__(self):
        super(Local_GNN, self).__init__()
        self._setup()
    def _setup(self):

        self.graph_convolution_1 = GCNConv(112,64) 
        self.graph_convolution_2 = GCNConv(64,20) 
        self.index_select_1 = SABP(20, ratio=0.9)
        self.graph_convolution_3 = GCNConv(20,20)

    def forward(self, data):
        edges, features = data.edge_index, data.x
        edges, features = edges.to(opt.device), features.to(opt.device)
        edge_attr = data.edge_attr
        edge_attr = edge_attr.to(opt.device).to(torch.float32)
        node_features_1 = torch.nn.functional.relu(self.graph_convolution_1(features, edges, edge_attr))
        node_features_2 = torch.nn.functional.relu(self.graph_convolution_2(node_features_1, edges, edge_attr))
        pool_features_2, pool_edge_index, pool_edge_attr, batch, perm, mi = self.index_select_1(node_features_2, edges, edge_attr,None)
        node_features_3 = torch.nn.functional.relu(self.graph_convolution_3(pool_features_2, pool_edge_index, pool_edge_attr))
        cat_feature = pool_features_2 + node_features_3
        graph_embedding = cat_feature.view(1, -1)
        return graph_embedding,perm, mi


class SABP(torch.nn.Module):
    def __init__(self,in_channels,ratio=0.8,Conv=GCNConv,non_linearity=torch.tanh):
        super(SABP,self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = Conv(in_channels,1)
        self.non_linearity = non_linearity
        self.gcn = Conv(in_channels,in_channels)
        self.fc = torch.nn.Linear(in_channels*2, 1)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        #x = x.unsqueeze(-1) if x.dim() == 1 else x
        score_neg = x[torch.randperm(x.size(0))] 
        embed = self.gcn(x,edge_index,edge_attr)
        joint = torch.cat((embed, x),dim = -1)
        margin = torch.cat((embed, score_neg),dim = -1)
        joint = self.fc(joint)
        margin = self.fc(margin)
        joint = F.normalize(joint, dim=1)
        margin = F.normalize(margin, dim=1)        
        mi_est = torch.mean(joint) - torch.log(torch.mean(torch.exp(margin)))    
        score = self.score_layer(x,edge_index,edge_attr).squeeze()
        perm = topk(score, self.ratio, batch)
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))
        return x, edge_index, edge_attr, batch, perm, mi_est

class Global_GNN(nn.Module):
    def __init__(self):
        super(Global_GNN, self).__init__()
        self.num_layers = 4
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(ChebConv(2020, 20,K=3,normalization='sym')) 
        self.bns.append(nn.BatchNorm1d(20))
        self.convs.append(ChebConv(20, 20,K=3,normalization='sym'))
        self.bns.append(nn.BatchNorm1d(20))
        self.convs.append(ChebConv(20, 20,K=3,normalization='sym'))
        self.bns.append(nn.BatchNorm1d(20))
        self.convs.append(ChebConv(20, 20,K=3,normalization='sym'))
        self.bns.append(nn.BatchNorm1d(20))
        self.out_fc = nn.Linear(20, 2)
        self.weights = torch.nn.Parameter(torch.randn((len(self.convs))))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.out_fc.reset_parameters()
        torch.nn.init.normal_(self.weights)

    def forward(self, features, edges, edge_weight):
        x = features 
        layer_out = []  
        x = self.convs[0](x, edges)
        x = self.bns[0](x)
        x = F.relu(x, inplace=True)
        layer_out.append(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.convs[1](x, edges)
        x = self.bns[1](x)
        x = F.relu(x, inplace=True)
        x = x +  0.7 * layer_out[0]
        layer_out.append(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.convs[2](x, edges)
        x = self.bns[2](x)
        x = F.relu(x, inplace=True)
        x = x +  0.7 * layer_out[1]
        layer_out.append(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.convs[3](x, edges)
        x = self.bns[3](x)
        x = F.relu(x, inplace=True)
        x = x +  0.7 * layer_out[2]
        layer_out.append(x)
        weight = F.softmax(self.weights, dim=0)
        for i in range(len(layer_out)):
            layer_out[i] = layer_out[i] * weight[i]
        emb = sum(layer_out)
        x = self.out_fc(emb)
        return x
class lg_gnn(torch.nn.Module):

    def __init__(self,nonimg, phonetic_score):
        super(lg_gnn, self).__init__()

        self.nonimg = nonimg
        self.phonetic_score = phonetic_score

        self._setup()

    def _setup(self):
        self.edge = EDGE(2, dropout= 0.2)  
        self.graph_level_model = Local_GNN()
        self.hierarchical_model = Global_GNN()
    def forward(self, graphs):
        dl = dataloader()
        embeddings = []
        h = 0
        perms = []
        MI = 0
        for graph in graphs:
            embedding,perm,mi= self.graph_level_model(graph)
            MI = MI + mi
            perm = perm.cpu().numpy()
            perms.append(perm)
            embeddings.append(embedding)
        embeddings = torch.cat(tuple(embeddings))
        mi_loss = MI/len(graphs)
        edge_index, edge_input = dl.get_inputs(self.nonimg, embeddings, self.phonetic_score) 
        edge_input = (edge_input- edge_input.mean(axis=0)) / edge_input.std(axis=0)
        edge_index = torch.tensor(edge_index, dtype=torch.long).to(opt.device)
        edge_input = torch.tensor(edge_input, dtype=torch.float32).to(opt.device)
        edge_weight = torch.squeeze(self.edge(edge_input))
        predictions = self.hierarchical_model(embeddings, edge_index, edge_weight)
        return predictions,mi_loss
