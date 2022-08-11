import numpy as np
import os
import pandas as pd
import math

import torch
import torch.nn as nn

from torch_geometric.loader import DataLoader,NeighborLoader
from torch_geometric.utils import to_undirected
    
ROOT_DIR = '/home/sandbox/workspace/sequence-graphs/data/'

def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

class AdversarialSampler:
    def __init__(self,edge_indices,node_indices,edge_sampling_values,K=10):
        
        self.edge_indices = edge_indices
        self.node_indices = node_indices
        self.edge_sampling_values = edge_sampling_values
        self.K = K
        
        self.get_candidate_edges()
        
        # create candidate edge sampler
        self.candidate_sampler = NeighborSampler(self.candidate_edge_indices,
                                                 sizes=[1])
    
    def get_candidate_edges(self):
        
        allneigh_sampler = NeighborSampler(self.edge_indices,sizes=[-1])
        
        e_id_list = []
        for node_idx in self.node_indices:
            out = allneigh_sampler.sample(node_idx.unsqueeze(0))
            n_neighbors = out[2].e_id.size(0)
            n_candidates = min(self.K,n_neighbors)
            inds2keep = torch.topk(self.edge_sampling_values[out[2].e_id],
                                   n_candidates).indices
            e_id_list.extend(out[2].e_id[inds2keep])
        
        self.candidate_edge_indices = self.edge_indices[:,e_id_list]
        self.candidate_e_id = torch.stack(e_id_list)
        
    def sample(self,node_indices):
        out = self.candidate_sampler.sample(node_indices)[2]
        return self.candidate_e_id[out.e_id]

class AdversarialSampler_v2:
    def __init__(self,edge_indices,node_indices,edge_sampling_values,K=10):
        
        self.edge_indices = edge_indices
        self.node_indices = node_indices
        self.edge_sampling_values = edge_sampling_values
        self.K = K
        
        self.get_candidate_edges()
        
        # create candidate edge sampler
        self.candidate_sampler = NeighborSampler(self.candidate_edge_indices,
                                                 sizes=[1])
    
    def get_candidate_edges(self):
        
        adj = SparseTensor(row=self.edge_indices[0],
                           col=self.edge_indices[1],
                           value=self.edge_sampling_values)
        
        allneigh_sampler = NeighborSampler(self.edge_indices,sizes=[-1])
        
        e_id_list = []
        for node_idx in self.node_indices:
            out = allneigh_sampler.sample(node_idx.unsqueeze(0))
            n_neighbors = out[2].e_id.size(0)
            n_candidates = min(self.K,n_neighbors)
            inds2keep = torch.topk(self.edge_sampling_values[out[2].e_id],
                                   n_candidates).indices
            e_id_list.extend(out[2].e_id[inds2keep])
        
        self.candidate_edge_indices = self.edge_indices[:,e_id_list]
        self.candidate_e_id = torch.stack(e_id_list)
        
    def sample(self,node_indices):
        out = self.candidate_sampler.sample(node_indices)[2]
        return self.candidate_e_id[out.e_id]
    
def load_ogb_splits(dataset):
    
    if 'ogb' in dataset:        
        if 'ogbn' in dataset:
            dataset_dir = os.path.join(ROOT_DIR,'ogb_npp',dataset)
        elif 'ogbg' in dataset:
            dataset_dir = os.path.join(ROOT_DIR,'ogb_gpp',dataset)

        train_idx = pd.read_csv(os.path.join(dataset_dir,'split','time','train.csv.gz'),
                               compression='gzip').values.squeeze()
        valid_idx = pd.read_csv(os.path.join(dataset_dir,'split','time','valid.csv.gz'),
                               compression='gzip').values.squeeze()
        test_idx = pd.read_csv(os.path.join(dataset_dir,'split','time','test.csv.gz'),
                              compression='gzip').values.squeeze()
        
    elif dataset in ['Cora','CiteSeer','PubMed']:
        
        dataset_dir = os.path.join(ROOT_DIR,'planetoid',dataset)
        data,_ = torch.load(os.path.join(dataset_dir,'processed',
                                       'data.pt'))
        
        train_idx = np.where(data.train_mask.data.numpy())[0]
        valid_idx = np.where(data.val_mask.data.numpy())[0]
        test_idx = np.where(data.test_mask.data.numpy())[0]
        
    train_idx = torch.LongTensor(train_idx)
    valid_idx = torch.LongTensor(valid_idx)
    test_idx = torch.LongTensor(test_idx)
    
    return train_idx,valid_idx,test_idx
    
def load_ogb_data(dataset):
        
    if 'ogb' in dataset:
        
        if 'ogbn' in dataset:
            dataset_dir = os.path.join(ROOT_DIR,'ogb_npp',dataset)
        elif 'ogbg' in dataset:
            dataset_dir = os.path.join(ROOT_DIR,'ogb_gpp',dataset)
        
        data,_ = torch.load(os.path.join(dataset_dir,'processed',
                                   'geometric_data_processed.pt'))
        
    elif dataset in ['Cora','CiteSeer','PubMed']:
        dataset_dir = os.path.join(ROOT_DIR,'planetoid',dataset)
        data,_ = torch.load(os.path.join(dataset_dir,'processed',
                                       'data.pt'))
        
    if 'arxiv' in dataset:
        
        X = data.x
        edge_indices = data.edge_index[[1,0]]
        Y = data.y
        
        # ensure edges point in direction of increasing time
        delta_year = data.node_year[edge_indices[0]]-data.node_year[edge_indices[1]]
        rev_idx = delta_year.squeeze() < 0
        edge_indices[:,rev_idx] = edge_indices[[1,0]][:,rev_idx]
        
        # positional encoding of paper years
        max_node_year = data.node_year.data.numpy().max()
        min_node_year = data.node_year.data.numpy().min()

        pos_encoding = positionalencoding1d(16,max_node_year-min_node_year+1)
        year_src_enc = max_node_year-data.node_year[data.edge_index[0]].squeeze()
        year_target_enc = max_node_year-data.node_year[data.edge_index[1]].squeeze()
        edge_attr = pos_encoding[year_src_enc]-pos_encoding[year_target_enc]

        pred_criterion = nn.CrossEntropyLoss(reduction='none')
        dim_in = data.x.shape[1]
        dim_out = 40
        
    elif dataset in ['Cora','CiteSeer','PubMed']:
        edge_attr = None
        edge_indices = data.edge_index[[1,0]]
        X = data.x
        Y = data.y
        pred_criterion = nn.CrossEntropyLoss(reduction='none')
        dim_in = data.x.shape[1]
        dim_out = Y.data.numpy().max()+1
        
    X = torch.FloatTensor(X)
    Y = torch.LongTensor(Y).squeeze()
    edge_indices = torch.LongTensor(edge_indices)
    edge_attr = None if edge_attr is None else torch.FloatTensor(edge_attr)
    edge_dim = None if edge_attr is None else edge_attr.shape[1]
    
    addl_params = {'dim_in': dim_in,
                   'dim_out': dim_out,
                   'edge_dim': edge_dim,
                   'pred_criterion': pred_criterion}
    
    return X,edge_indices,Y,edge_attr,addl_params

def load_dataloader(dataset_name,batch_size=256,shuffle_train=True):
                
    if 'ogbn' in dataset_name:

        from ogb.nodeproppred import PygNodePropPredDataset

        dataset = PygNodePropPredDataset(name = dataset_name, 
                                          root = os.path.join(ROOT_DIR,'ogb_npp'))
        split_idx = dataset.get_idx_split()
        
        if 'arxiv' in dataset_name:

            # positional encoding of paper years
            max_node_year = dataset.data.node_year.data.numpy().max()
            min_node_year = dataset.data.node_year.data.numpy().min()
        
            pos_encoding = positionalencoding1d(16,max_node_year-min_node_year+1)
            year_src_enc = max_node_year-dataset.data.node_year[dataset.data.edge_index[0]].squeeze()
            year_target_enc = max_node_year-dataset.data.node_year[dataset.data.edge_index[1]].squeeze()
            dataset.data.edge_attr = pos_encoding[year_src_enc]-pos_encoding[year_target_enc]
            dataset.data.y = dataset.data.y.squeeze()
            
        train_loader = NeighborLoader(dataset.data,num_neighbors=[-1], 
                                      input_nodes=split_idx['train'], 
                                      batch_size=batch_size,shuffle=shuffle_train)
        valid_loader = NeighborLoader(dataset.data,num_neighbors=[-1],
                                      input_nodes=split_idx['valid'],
                                      batch_size=batch_size,shuffle=False)
        test_loader = NeighborLoader(dataset.data,num_neighbors=[-1],
                                     input_nodes=split_idx['test'],
                                     batch_size=batch_size,shuffle=False)
        
    elif dataset_name in ['Cora','CiteSeer','PubMed']:
        
        dataset_dir = os.path.join(ROOT_DIR,'planetoid',dataset_name)
        data,_ = torch.load(os.path.join(dataset_dir,'processed','data.pt'))
        data.y = data.y #.unsqueeze(-1).float()
        
        train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
        valid_idx = data.val_mask.nonzero(as_tuple=False).view(-1)
        test_idx = data.test_mask.nonzero(as_tuple=False).view(-1)

        train_loader = NeighborLoader(data,num_neighbors=[-1],input_nodes=train_idx, 
                                      batch_size=batch_size,shuffle=True)
        valid_loader = NeighborLoader(data,num_neighbors=[-1],input_nodes=valid_idx,
                                      batch_size=batch_size,shuffle=False)
        test_loader = NeighborLoader(data,num_neighbors=[-1],input_nodes=test_idx,
                                     batch_size=batch_size,shuffle=False)
        
    elif 'ogbg' in dataset_name:
        
        from ogb.graphproppred import PygGraphPropPredDataset
        
        dataset = PygGraphPropPredDataset(name = dataset_name, 
                                          root = os.path.join(ROOT_DIR,'ogb_gpp'))
        dataset.data.y = dataset.data.y.float()

        split_idx = dataset.get_idx_split() 
        train_loader = DataLoader(dataset[split_idx["train"]], 
                                  batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(dataset[split_idx["valid"]], 
                                  batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset[split_idx["test"]], 
                                 batch_size=batch_size, shuffle=False)
    
    elif 'ogbl' in dataset_name:

        from ogb.linkproppred import PygLinkPropPredDataset

        dataset = PygLinkPropPredDataset(name = dataset_name,
                                         root = os.path.join(ROOT_DIR,'ogb_lpp'))
        dataset.data.n_id = torch.arange(dataset.data.num_nodes)
        
        if dataset_name == 'ogbl-ddi':
            dataset.data.x = dataset.data.n_id.unsqueeze(1) #.copy()
            
#         if dataset_name == 'ogbl-collab':
#             # positional encoding of paper years
#             max_edge_year = dataset.data.edge_year.data.numpy().max()
#             min_edge_year = dataset.data.edge_year.data.numpy().min()
        
#             pos_encoding = positionalencoding1d(16,max_edge_year-min_edge_year+1)
#             edge_year_encoding = pos_encoding[dataset.data.edge_year.squeeze()-min_edge_year]
#             dataset.data.edge_attr = torch.cat([edge_year_encoding,dataset.data.edge_weight],dim=1)
        
        split_idx = dataset.get_edge_split()

        train_dataset = dataset.copy()
        train_dataset.data.edge_index = split_idx['train']['edge'].T

        valid_dataset = dataset.copy()
        valid_pos_edges = split_idx['valid']['edge']
        valid_neg_edges = split_idx['valid']['edge_neg']

        valid_dataset.data.edge_index = split_idx['train']['edge'].T
        valid_dataset.data.edge_label_index = torch.cat([valid_pos_edges,
                                                   valid_neg_edges],dim=0).T
        valid_dataset.data.edge_label = torch.cat([torch.ones(valid_pos_edges.size(0)),
                                          torch.zeros(valid_neg_edges.size(0))])

        test_dataset = dataset.copy()
        test_pos_edges = split_idx['test']['edge']
        test_neg_edges = split_idx['test']['edge_neg']

        test_dataset.data.edge_index = split_idx['train']['edge'].T
        test_dataset.data.edge_label_index = torch.cat([test_pos_edges,
                                                   test_neg_edges],dim=0).T
        test_dataset.data.edge_label = torch.cat([torch.ones(test_pos_edges.size(0)),
                                          torch.zeros(test_neg_edges.size(0))])

        train_loader = NeighborLoader(train_dataset.data,
                                      num_neighbors=[-1],
                                      input_nodes=torch.unique(train_dataset.data.edge_index.flatten()),
                                      batch_size=batch_size,
                                      shuffle=True)

        valid_loader = NeighborLoader(valid_dataset.data,
                                      num_neighbors=[-1],
                                      input_nodes=torch.unique(valid_dataset.data.edge_index.flatten()),
                                      batch_size=batch_size,
                                      shuffle=False)

        test_loader = NeighborLoader(test_dataset.data,
                                      num_neighbors=[-1],
                                      input_nodes=torch.unique(test_dataset.data.edge_index.flatten()),
                                      batch_size=batch_size,
                                      shuffle=False)

    return train_loader,valid_loader,test_loader

def aggregate_using_ptr(data,ptr,op='sum'):
    
    if op == 'mean':
        return torch.stack([data[ptr[i]:ptr[i+1]].mean(0) 
                        for i in range(ptr.size(0)-1)])
    elif op == 'sum':
        return torch.stack([data[ptr[i]:ptr[i+1]].sum(0) 
                        for i in range(ptr.size(0)-1)])
    elif op == 'median':
        return torch.stack([data[ptr[i]:ptr[i+1]].median(0).values 
                        for i in range(ptr.size(0)-1)])

def get_dataset_params(dataset_name,dataloader,dim_hidden):
    
    if 'ogbg-mol' in dataset_name:
        dim_in = dim_hidden
        dim_out = dataloader.dataset.data.y.shape[1]
        edge_dim = dim_hidden
        
        if any(x in dataset_name for x in ['molesol','molfreesolv','mollipo']):
            pred_criterion = nn.MSELoss(reduction='none')
        else:
            pred_criterion = nn.BCELoss(reduction='none')
    
    elif 'ogbn' in dataset_name or dataset_name in ['Cora','CiteSeer','PubMed']:
        dim_in = dataloader.data.x.shape[1]
        dim_out = dataloader.data.y.long().data.numpy().max()+1
        edge_dim = dataloader.data.edge_attr.shape[1] if dataloader.data.edge_attr is not None else None
        pred_criterion = nn.CrossEntropyLoss(reduction='none')
    
    elif 'ogbl' in dataset_name:
        dim_in = dim_hidden if dataset_name == 'ogbl-ddi' else dataloader.data.x.shape[1]
        dim_out = 1
        edge_dim = dataloader.data.edge_attr.shape[1] if dataloader.data.edge_attr is not None else None
        pred_criterion = nn.BCELoss(reduction='none')
        
    return dim_in,dim_out,edge_dim,pred_criterion

def generate_rewired_dataloader(dataloader,e_id,batch_size=256,shuffle=True):
    
    data = dataloader.data
    data.edge_index = data.edge_index[:,e_id]
    data.edge_attr = None if data.edge_attr is None else train_data.edge_attr[:,e_id]

    return NeighborLoader(data,num_neighbors=[-1],batch_size=batch_size,shuffle=shuffle)