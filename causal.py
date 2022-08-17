import numpy as np
import time
from scipy.sparse import coo_array

import torch
import torch.nn as nn
from torch_geometric.loader import NeighborSampler
from torch_sparse import SparseTensor

from utils import *

def get_pruned_e_id(edge_indices,pruned_edge_indices):
    
    pruned_edges = set([(n[0],n[1]) for n 
                        in pruned_edge_indices.T.data.numpy()])
    return torch.LongTensor(np.array([i for i,n in enumerate(edge_indices.cpu().T.data.numpy()) 
               if (n[0],n[1]) in pruned_edges])).to(edge_indices.device)

def select_edges_to_prune(node_indices,edge_indices,
                          sampler=None,
                          edge_attr=None,sampling='uniform',
                          candidate_edge_indices=None):
    
    # sample edges to prune
    if sampling == 'uniform':
        pruned_e_id = sampler.sample(node_indices)[2].e_id

    elif sampling == 'adversarial':
        # pruned_e_id = sampler.sample(node_indices)
        pruned_e_id = get_pruned_e_id(edge_indices,candidate_edge_indices)
        
    # generate mask of pruned edges
    mask = torch.ones(edge_indices.size(1),dtype=torch.bool, 
                      device=edge_indices.device)
    mask[pruned_e_id] = False

    return pruned_e_id,mask

def prune_edges(edge_indices,edge_attr,pruned_e_id,mask):
        
    remaining_edge_indices = edge_indices[:,mask]
    remaining_edge_attr = None if edge_attr is None else edge_attr[mask]
    
    return remaining_edge_indices,remaining_edge_attr
        
def create_edge_item_mapping(ptr,edge_indices):
    
    item_index = torch.cat([torch.ones(ptr[i+1]-ptr[i])*i 
                          for i in range(ptr.size(0)-1)])
    node_index = torch.arange(ptr.max())

    edge_item_mapping = item_index[edge_indices[0]]

    return torch.stack([torch.arange(edge_indices.size(1)),
                                  edge_item_mapping]).long()

def compute_causal_effect(model,X,Y,preds,remaining_edge_indices,
                          edge_attr,node_indices,pred_criterion,
                          task='npp',ptr=None,edge_indices_pred=None,
                          loss_ratio=False):
    
    if task == 'npp':
        pred_loss = pred_criterion(preds[node_indices],
                               Y[node_indices])
        interv_preds,_ = model(X,remaining_edge_indices,edge_attr=edge_attr)
        interv_pred_loss = pred_criterion(interv_preds[node_indices],
                                      Y[node_indices])
    elif task == 'gpp':
        pred_loss = pred_criterion(preds[node_indices],
                               Y[node_indices])
        interv_preds,_ = model(X,remaining_edge_indices,ptr,edge_attr)
        interv_pred_loss = pred_criterion(interv_preds[node_indices],
                                      Y[node_indices])
    elif task == 'lpp':
        
        num_nodes = max(edge_indices_pred.max() + 1,node_indices.max() + 1)
        
        # loss per node (original)
        pred_loss = pred_criterion(preds,Y)
        loss_adj = SparseTensor(row=edge_indices_pred[0],col=edge_indices_pred[1],
                                value=pred_loss,sparse_sizes=(num_nodes, num_nodes))
        pred_loss = loss_adj.mean(1)[node_indices]

        # loss per node (intervened)
        interv_preds,_ = model(X,remaining_edge_indices,edge_indices_pred,edge_attr)
        interv_pred_loss = pred_criterion(interv_preds,Y)
        interv_loss_adj = SparseTensor(row=edge_indices_pred[0],col=edge_indices_pred[1],
                                value=interv_pred_loss,sparse_sizes=(num_nodes, num_nodes))
        interv_pred_loss = loss_adj.mean(1)[node_indices]
        
    if loss_ratio:
        causal_effect = interv_pred_loss/(1e-20 + pred_loss)
        causal_effect = 1/(1+torch.exp(-10*(causal_effect-1)))
    else:
        causal_effect = interv_pred_loss - pred_loss #/(1e-20 + pred_loss)
        causal_effect = 1/(1+torch.exp(-1*(causal_effect-1)))
    causal_effect = causal_effect.detach()
    
    # # lpp task (causal effect matrix)
    # # print(causal_effect.shape)
    # if causal_effect.dim() > 1:
    #     print(node_indices.shape,causal_effect.shape)
    #     causal_effect = causal_effect.mean(1)[node_indices]
    #     print(causal_effect.shape)
    return causal_effect.squeeze()

def identify_candidate_edges(node_indices,edge_indices,edge_sampling_values):
    
    num_nodes = edge_indices.max() + 1
    adj = coo_array((edge_sampling_values, edge_indices), (num_nodes,num_nodes))

    i = np.array(adj.argmax(0)).squeeze()[node_indices]
    j = node_indices

    return torch.LongTensor(np.array([i,j]))
                
def compute_intervention_loss(model,X,node_indices,edge_indices,Y,preds,
                              attn_weights_list,device=0,n_interventions_per_node=10,
                              edge_attr=None,
                              pred_criterion=nn.BCELoss(),
                              sampling='uniform',edge_sampling_values=None,
                              weight_by_degree=False,
                              task='npp',ptr=None,edge_indices_pred=None,
                              loss_ratio=False):
    
    sigmoid = nn.Sigmoid()
    
    if task == 'npp' or task == 'lpp':
        if sampling == 'uniform':
            # select one edge per node
            sampler = NeighborSampler(edge_indices,
                                      sizes=[1],num_nodes=X.shape[0])
            candidate_edge_indices = None
        elif sampling == 'adversarial':
            # identify highest value per node
            sampler = None
            candidate_edge_indices = identify_candidate_edges(node_indices.cpu().data.numpy(),
                                                       edge_indices.cpu().data.numpy(),
                                                       edge_sampling_values.cpu().data.numpy())
    elif task == 'gpp':
        num_nodes = ptr.max()
        edge_item_indices = create_edge_item_mapping(ptr,edge_indices)
        sampler = NeighborSampler(edge_item_indices,sizes=[1])
        candidate_edge_indices = None
                
    causal_interv_loss = 0
    for intervention_no in range(n_interventions_per_node):
        
        # select & prune edges
        pruned_e_id,mask = select_edges_to_prune(node_indices,edge_indices,
                                                 sampler=sampler,edge_attr=edge_attr,
                                                 sampling=sampling,
                                                 candidate_edge_indices=candidate_edge_indices)
        remaining_edge_indices,remaining_edge_attr = prune_edges(edge_indices,edge_attr,
                                                                 pruned_e_id,mask)
    
        if task == 'gpp':
            nodes_to_evaluate = node_indices
        elif task == 'npp' or task == 'lpp':
            nodes_to_evaluate = edge_indices[1,pruned_e_id]
        
        # compute causal effect
        causal_effect = compute_causal_effect(model,X,Y,preds,remaining_edge_indices,
                                              remaining_edge_attr,nodes_to_evaluate,
                                              pred_criterion,task=task,ptr=ptr,
                                              edge_indices_pred=edge_indices_pred,
                                              loss_ratio=loss_ratio)
        
        # weight loss function
        if weight_by_degree:
            from collections import Counter
            
            counts = Counter(edge_indices.data.numpy()[1])
            w = np.array([counts[n] for n in nodes_with_pruned_parents.data.numpy()])
            w = torch.FloatTensor(w).to(edge_indices.device)
            
            bce_loss = nn.BCELoss(weight=w)
        else:
            bce_loss = nn.BCELoss()
            
        n_attn_layers = len(attn_weights_list)
        for attn_weights in attn_weights_list:
            
            # select attention weights (orig model) related to intervened edges
            attn_weights_intervened_edge = attn_weights[pruned_e_id].mean(1)
            
            interv_loss = bce_loss(attn_weights_intervened_edge,causal_effect)
            causal_interv_loss += interv_loss/n_interventions_per_node/n_attn_layers
        
    return causal_interv_loss