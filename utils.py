import numpy as np
import os
import pandas as pd
from typing import Optional, Tuple
import math

import torch
import torch.nn as nn

from torch_geometric.typing import OptTensor
from torch_geometric.utils.dropout import filter_adj

def convert_adjacency_to_dag(A,orient_by='in_degree'):
    """
    Converts the adjacency matrix A into a DAG by sorting nodes by their
    weighted out-degree or PageRank importance scores and retaining edges 
    that cohere with this sorting. Nodes are sorted by lowest to highest 
    values, so nodes with higher values lie upstream in the DAG.
    """
    
    # lower to higher
    if orient_by == 'out_degree':
        node_values = A.sum(1)
    elif orient_by == 'in_degree':
        node_values = A.sum(0)
    elif orient_by == 'pagerank':
        from sknetwork.ranking import PageRank
        
        pagerank = PageRank()
        node_values = pagerank.fit_transform(A)
        
    return A*(node_values - node_values[:,None] > 0).astype(float)
    
def normalize_adjacency(A,by='incoming'):
    """
    Normalizes the weights of the adjacency matrix A by the sum of the 
    weights of nodes' incoming or outgoing edges.
    Here, A[i,j] represents the weight of the edge going from node i to
    node j.
    """
    
    if by == 'outgoing':
        indegree = A.sum(1)
        indegree[indegree == 0] = 1
        norm_A = (A.T / indegree).T
    elif by == 'incoming':
        indegree = A.sum(0)
        indegree[indegree == 0] = 1
        norm_A = A/indegree
    
    return norm_A
  
def add_diagonal_and_renormalize(A,lam=0.1,by='incoming'):
    
    A_copy = A.copy()
    if by == 'incoming':
        A_copy += lam*np.eye(A.shape[0])
        
    return normalize_adjacency(A_copy)

def to_sparse_coo_tensor(A):
    
    # add small diagonal term if zero incoming/outgoing edges
    for i in np.where(A.sum(0) == 0)[0]:
        A[i,i] = 1e-10
    for i in np.where(A.sum(1) == 0)[0]:
        A[i,i] = 1e-10
        
    from scipy.sparse import coo_matrix
    
    sp_A = coo_matrix(A)
    idx = torch.FloatTensor(np.array([sp_A.row,sp_A.col]))
    data = torch.FloatTensor(sp_A.data)
    
    return torch.sparse_coo_tensor(idx,data)

def construct_dag_from_sequence(L):
	
	dag = np.zeros((L,L))
	for i in range(L-1):
		dag[i+1,i] = 1
	return dag

def create_AnX_list(S,S_1,X,L):
    
	AnX_list = [S @ X]
	for n in range(1,L):
		AnX_list.append(S_1 @ AnX_list[-1])
	return AnX_list

def model_forecast(model,X,S,S_1,L):
		
	AnX_list = create_AnX_list(S,S_1,X,L)
	return model(AnX_list)[-1].cpu().data.numpy()

def model_forecast_embed(model,X,S,S_1):
	
	return model(X,S,S_1)[-1].cpu().data.numpy()

def populate_embeddings_df_with_consumers(embeddings_df,consumer_embedding_dict):
    
    newrows_dict = {k: [] for k in embeddings_df.columns}
    prod_ids = set(embeddings_df['user_id'])

    for i,source_id in enumerate(sorted(list(consumer_embedding_dict.keys()))):
        newrows_dict['user_id'].append(source_id)
        newrows_dict['embedding'].append(consumer_embedding_dict[source_id])
        newrows_dict['idx'].append(i + embeddings_df['idx'].values.max() + 1)
        newrows_dict['producer'].append(int(source_id) in prod_ids)
        newrows_dict['consumer'].append(True)

    cons_embeddings_df = pd.concat([embeddings_df,pd.DataFrame(newrows_dict)])
    cons_embeddings_df['user_id'] = cons_embeddings_df['user_id'].astype(int)

    return cons_embeddings_df

class customLoss(nn.Module):
    def __init__(self):
        super(customLoss, self).__init__()
        
        print('using custom loss')
        self.cs = nn.CosineSimilarity()
        self.mse = nn.MSELoss(reduction='none')
    
    def forward(self,output,target):
        
        return -self.cs(output,target) + self.mse(output,target).sum(1)
    
class NegativeCosineSimilarity(nn.Module):
    def __init__(self):
        super(NegativeCosineSimilarity, self).__init__()
        
        self.cs = nn.CosineSimilarity()
    
    def forward(self,output,target):
        
        return -self.cs(output,target)
    
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

ROOT_DIR = '/home/sandbox/workspace/sequence-graphs/data/'

def load_ogb_splits(dataset):
    
    if dataset in ['arxiv','mag']:
        dataset_dir = os.path.join(ROOT_DIR,'ogb_npp','ogbn_{}'.format(dataset))
        
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
        
    return train_idx,valid_idx,test_idx
    
def load_ogb_data(dataset):
        
    if dataset in ['arxiv','mag']:
        dataset_dir = os.path.join(ROOT_DIR,'ogb_npp','ogbn_{}'.format(dataset))
        data,_ = torch.load(os.path.join(dataset_dir,'processed',
                                       'geometric_data_processed.pt'))
        
    elif dataset in ['Cora','CiteSeer','PubMed']:
        dataset_dir = os.path.join(ROOT_DIR,'planetoid',dataset)
        data,_ = torch.load(os.path.join(dataset_dir,'processed',
                                       'data.pt'))
        
    if dataset == 'arxiv':
        
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
        edge_loss_weights = None
        dim_in = data.x.shape[1]
        dim_out = 40
        
    elif dataset in ['Cora','CiteSeer','PubMed']:
        edge_attr = None
        edge_indices = data.edge_index[[1,0]]
        X = data.x
        Y = data.y
        pred_criterion = nn.CrossEntropyLoss(reduction='none')
        edge_loss_weights = None
        dim_in = data.x.shape[1]
        dim_out = Y.data.numpy().max()+1
        
    X = torch.FloatTensor(X)
    Y = torch.LongTensor(Y).squeeze()
    edge_indices = torch.LongTensor(edge_indices)
    edge_attr = None if edge_attr is None else torch.FloatTensor(edge_attr)
    edge_dim = None if edge_attr is None else edge_attr.shape[1]
    
    addl_params = {'edge_loss_weights': edge_loss_weights,
              'dim_in': dim_in,
              'dim_out': dim_out,
              'edge_dim': edge_dim,
              'pred_criterion': pred_criterion}
    
    return X,edge_indices,Y,edge_attr,addl_params