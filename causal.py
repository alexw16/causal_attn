import numpy as np
import time

import torch
import torch.nn as nn
from torch_geometric.loader import NeighborSampler
from torch_sparse import SparseTensor

from utils import *

def compute_intervention_loss_pp(model,X,node_indices,parent_target_indices,target_child_indices,
                              pred_loss,attn_weights,device=0,n_interventions_per_node=3,
                              edge_attr=None,edge_loss_weights=None,
                              pred_criterion=NegativeCosineSimilarity()):

    neighIntervSampler = NeighborSampler(parent_target_indices,sizes=[1],
                                         num_nodes=X.shape[0])
    
    causal_interv_loss = 0
    for intervention_no in range(n_interventions_per_node):
        
        # sample edges to prune + prune edges
        out = neighIntervSampler.sample(node_indices)
        pruned_e_id = out[2].e_id
        mask = torch.ones(parent_target_indices.size(1),dtype=torch.bool, 
                          device=parent_target_indices.device)
        mask[pruned_e_id] = False
        remaining_edge_indices = parent_target_indices[:,mask]

        # forecasting loss (original vs. intervened)
        sampled_edge_attr = None if edge_attr is None else edge_attr[mask].to(device)

        interv_preds,_ = model(X,remaining_edge_indices,edge_attr=sampled_edge_attr)
        interv_pred_loss = pred_criterion(interv_preds[target_child_indices[0]],
                                          X[target_child_indices[1]])
        delta_loss = interv_pred_loss - pred_loss

        # forecasting loss per target node
        delta_loss_adj = SparseTensor(row=target_child_indices[0],
                                      col=target_child_indices[1],
                                      value=delta_loss)
        delta_loss_per_node_mean = delta_loss_adj.mean(1)

        # select attention weights (orig model) related to intervened edges
        attn_weights_intervened_edge = attn_weights[pruned_e_id].mean(1)

        # identify nodes with pruned parents
        nodes_with_pruned_parents = parent_target_indices[:,pruned_e_id][1]
        delta_loss_per_node_mean = delta_loss_per_node_mean[nodes_with_pruned_parents]

        interv_loss = -attn_weights_intervened_edge.dot(delta_loss_per_node_mean)
        interv_loss = interv_loss/nodes_with_pruned_parents.size(0)/n_interventions_per_node
        causal_interv_loss += interv_loss
        
    return causal_interv_loss

def compute_intervention_loss_ogb(model,X,node_indices,parent_target_indices,Y,preds,
                              attn_weights_list,device=0,n_interventions_per_node=10,
                              edge_attr=None,edge_loss_weights=None,
                              pred_criterion=NegativeCosineSimilarity(),
                              sampling='uniform',edge_sampling_values=None,
                              weight_by_degree=False):
    
    sigmoid = nn.Sigmoid()
    relu = nn.ReLU()
    
    if sampling == 'uniform':
        # select one edge per node
        sampler = NeighborSampler(parent_target_indices,sizes=[1],
                                             num_nodes=X.shape[0])
    elif sampling == 'adversarial':
        # get all node parents
        sampler = NeighborSampler(parent_target_indices,sizes=[-1],
                                             num_nodes=X.shape[0])

    causal_interv_loss = 0
    for intervention_no in range(n_interventions_per_node):
        
        # sample edges to prune
        if sampling == 'uniform':
            out = sampler.sample(node_indices)
            pruned_e_id = out[2].e_id
            
        elif sampling == 'adversarial':
            pruned_e_id = get_top_K_per_node(sampler,node_indices,
                                             parent_target_indices,
                                             edge_sampling_values,1)
        
        # identify nodes with pruned parents
        nodes_with_pruned_parents = parent_target_indices[1,pruned_e_id]
        
        # weight loss function
        if weight_by_degree:
            from collections import Counter
            
            counts = Counter(parent_target_indices.data.numpy()[1])
            w = np.array([counts[n] for n in nodes_with_pruned_parents.data.numpy()])
            w = torch.FloatTensor(w).to(parent_target_indices.device)
            
            bce_loss = nn.BCELoss(weight=w)
        else:
            bce_loss = nn.BCELoss()
    
        # get remaining edges
        mask = torch.ones(parent_target_indices.size(1),dtype=torch.bool, 
                          device=parent_target_indices.device)
        mask[pruned_e_id] = False
        remaining_edge_indices = parent_target_indices[:,mask]

        # forecasting loss (original vs. intervened)
        sampled_edge_attr = None if edge_attr is None else edge_attr[mask].to(device)

        interv_preds,_ = model(X,remaining_edge_indices,edge_attr=sampled_edge_attr)

        pred_loss = pred_criterion(preds[nodes_with_pruned_parents],
                                   Y[nodes_with_pruned_parents])
        interv_pred_loss = pred_criterion(interv_preds[nodes_with_pruned_parents],
                                          Y[nodes_with_pruned_parents])

        causal_effect = interv_pred_loss - pred_loss
        causal_effect = 1/(1+torch.exp(-1*(causal_effect-1)))
        causal_effect = causal_effect.detach()
        
        n_attn_layers = len(attn_weights_list)
        for attn_weights in attn_weights_list:
            
            # select attention weights (orig model) related to intervened edges
            attn_weights_intervened_edge = attn_weights[pruned_e_id].mean(1)

            interv_loss = bce_loss(attn_weights_intervened_edge,causal_effect)
            causal_interv_loss += interv_loss/n_interventions_per_node/n_attn_layers
        
    return causal_interv_loss

def run_epoch_batch(epoch_no,model,X,edge_indices,Y,model_type='causal',optimizer=None,
                    node_indices=None,
                    edge_attr=None,device=0,batch_size=15000,verbose=True,
                    train=True,pred_criterion=NegativeCosineSimilarity(),
                    edge_loss_weights=None,
                    direction_loss=False,intervention_loss=False,
                    lam_causal=1):
        
    np.random.seed(epoch_no)
    torch.manual_seed(epoch_no)
    
    start = time.time()
    epoch_loss = 0
    
    model = model.to(device)
    X = X.to(device)
    Y = Y.to(device)
    
    neighSampler = NeighborSampler(edge_indices,sizes=[-1],
                                   num_nodes=X.shape[0])
    neighSampler_rev = NeighborSampler(edge_indices[[1,0]],sizes=[-1],
                                       num_nodes=X.shape[0])
    
    weight_by_degree = 'wbd' in model_type
    
    if node_indices is None:
        node_indices = torch.arange(X.shape[0])
    
    if train:
        # permute node indices if training
        node_indices = node_indices[torch.randperm(node_indices.shape[0])]
    
    preds_list = []
    for i in range(0,node_indices.shape[0],batch_size):

        # identify parent + child nodes of node indices
        out = neighSampler.sample(node_indices[i:i+batch_size])
        parent_target_indices = edge_indices[:,out[2].e_id].to(device)
                
        sampled_edge_attr = None if edge_attr is None else edge_attr[out[2].e_id].to(device)
        
        # node-level predictions
        preds,attn_weights = model(X,parent_target_indices,sampled_edge_attr)
        
        if train:
            
            # prediction loss
            pred_loss = pred_criterion(preds[node_indices[i:i+batch_size]],
                                       Y[node_indices[i:i+batch_size]]).mean()
            
            # causal direction loss
            causal_dir_loss = 0
            if direction_loss:
                continue
            
            # causal intervention loss
            causal_interv_loss = 0
            if intervention_loss:
                causal_interv_loss = compute_intervention_loss_ogb(
                                        model,X,node_indices[i:i+batch_size],
                                        parent_target_indices,Y,preds,attn_weights,
                                        device=device,edge_attr=sampled_edge_attr,
                                        pred_criterion=pred_criterion,
                                        weight_by_degree=weight_by_degree)
            
            total_loss = pred_loss + lam_causal*causal_interv_loss + causal_dir_loss
            
            if 'adversarial' in model_type:
                 attn_weights.retain_grad()
                
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if 'adversarial' in model_type:
                attn_grads = attn_weights.grad.mean(1)
            
            epoch_loss += total_loss.cpu().data.numpy()
            
        else:
            preds_list.append(preds[node_indices[i:i+batch_size]].detach())
                    
    end = time.time()
    
    if verbose and epoch_no % 5 == 0:
        mode = 'train' if train else 'test'
        print_str = 'Epoch {} ({:.5f} seconds)'.format(
            epoch_no,end-start)
        if train:
            print_str += ': total loss {:.5f}, pred loss {:.5f}'.format(epoch_loss,
                                                                        pred_loss.cpu().data.numpy())
            if direction_loss:
                print_str += ', dir loss {:.5f}'.format(causal_dir_loss.cpu().data.numpy())
            if intervention_loss:
                print_str += ', interv loss {:.5f}'.format(causal_interv_loss.cpu().data.numpy())
        print(print_str)
    
    if not train:
        preds = torch.cat(preds_list)
        
    return epoch_loss,preds

def train_model(model,X,edge_indices,Y,model_type,optimizer,device,
                node_indices=None,
                edge_attr=None,num_epochs=10,edge_loss_weights=None,
                direction_loss=False,intervention_loss=False,
                lam_causal=1,
                early_stop=True,pred_criterion=NegativeCosineSimilarity(),tol=1e-5,verbose=True):
    
    past_loss = 1e10
    for epoch_no in range(num_epochs):
         
        current_loss,_ = run_epoch_batch(epoch_no,model,X,edge_indices,Y,model_type,optimizer,
                                         node_indices=node_indices,
                                         edge_attr=edge_attr,device=device,train=True,
                                         pred_criterion=pred_criterion,
                                         edge_loss_weights=edge_loss_weights,
                                         verbose=verbose,direction_loss=direction_loss,
                                         intervention_loss=intervention_loss,
                                         lam_causal=lam_causal)
        
        if early_stop and abs(past_loss - current_loss)/abs(past_loss) < tol:
            break
            
        past_loss = current_loss