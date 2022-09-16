import numpy as np
import time

import torch
import torch.nn as nn
from torch_geometric.utils import negative_sampling
from torch_geometric.transforms import RandomLinkSplit

from torch_geometric.loader import NeighborSampler

from utils import *
from causal import *




# def run_epoch_batch(epoch_no,model,X,edge_indices,Y,model_type='causal',
#                     optimizer=None,node_indices=None,edge_attr=None,device=0,
#                     batch_size=15000,verbose=True,train=True,
#                     pred_criterion=nn.BCELoss(),
#                     intervention_loss=False,
#                     lam_causal=1,n_interventions_per_node=10):
        
#     np.random.seed(epoch_no)
#     torch.manual_seed(epoch_no)
    
#     start = time.time()
#     total_loss = 0
#     total_pred_loss = 0
#     total_intervention_loss = 0
    
#     model = model.to(device)
#     X = X.to(device)
#     Y = Y.to(device)
    
#     neighSampler = NeighborSampler(edge_indices,sizes=[-1],
#                                    num_nodes=X.shape[0])

#     weight_by_degree = 'wbd' in model_type
    
#     if node_indices is None:
#         node_indices = torch.arange(X.shape[0])
    
#     if train:
#         # permute node indices if training
#         node_indices = node_indices[torch.randperm(node_indices.shape[0])]
    
#     preds_list = []
#     for i in range(0,node_indices.shape[0],batch_size):

#         # identify parent + child nodes of node indices
#         out = neighSampler.sample(node_indices[i:i+batch_size])
#         batch_edge_indices = edge_indices[:,out[2].e_id].to(device)
#         batch_edge_attr = None if edge_attr is None else edge_attr[out[2].e_id].to(device)
        
#         # node-level predictions
#         preds,attn_weights = model(X,batch_edge_indices,batch_edge_attr)
        
#         if train:
            
#             # prediction loss
#             pred_loss = pred_criterion(preds[node_indices[i:i+batch_size]],
#                                        Y[node_indices[i:i+batch_size]]).mean()

#             if 'adversarial' in model_type:
#                 sampling = 'adversarial'
#                 edge_sampling_values = torch.stack([attn.mean(1) for attn 
#                                                    in attn_weights]).mean(0)
                
#             else:
#                 sampling = 'uniform'
#                 edge_sampling_values = None
#                 # attn_weights.retain_grad()
                
#             # causal intervention loss
#             causal_interv_loss = 0
#             loss_ratio = 'ratio' in model_type
#             if intervention_loss:
#                 causal_interv_loss = compute_intervention_loss(
#                                         model,X,node_indices[i:i+batch_size],
#                                         batch_edge_indices,Y,preds,attn_weights,
#                                         device=device,edge_attr=batch_edge_attr,
#                                         pred_criterion=pred_criterion,sampling=sampling,
#                                         edge_sampling_values=edge_sampling_values,
#                                         weight_by_degree=weight_by_degree,
#                                         n_interventions_per_node=n_interventions_per_node,
#                                         loss_ratio=loss_ratio)
            
#             batch_loss = pred_loss + lam_causal*causal_interv_loss #+ causal_dir_loss
                
#             optimizer.zero_grad()
#             batch_loss.backward()
#             optimizer.step()
            
#             # if 'adversarial' in model_type:
#             #     attn_grads = attn_weights.grad.mean(1)
            
#             total_loss += batch_loss.cpu().data.numpy()
#             total_pred_loss += pred_loss.cpu().data.numpy()
#             total_intervention_loss += lam_causal*causal_interv_loss
            
#         else:
#             preds_list.append(preds[node_indices[i:i+batch_size]].detach())
                    
#     end = time.time()
    
#     if verbose and epoch_no % 5 == 0:
#         mode = 'train' if train else 'test'
#         print_str = 'Epoch {} ({:.5f} seconds)'.format(
#             epoch_no,end-start)
#         if train:
#             print_str += ': total loss {:.5f}, pred loss {:.5f}'.format(total_loss,
#                                                                         total_pred_loss)
#             if intervention_loss:
#                 print_str += ', interv loss {:.5f}'.format(total_intervention_loss)
#         print(print_str)
        
#     loss_dict = {'total_loss': total_loss,
#                  'pred_loss': total_pred_loss,
#                  'intervention loss': total_intervention_loss
#                  }
#     preds = None if train else torch.cat(preds_list)
    
#     return loss_dict,preds

# def train_model(model,X,edge_indices,Y,model_type,optimizer,device,
#                 node_indices=None,edge_attr=None,num_epochs=10,
#                 intervention_loss=False,
#                 lam_causal=1,n_interventions_per_node=10,early_stop=True,
#                 pred_criterion=nn.BCELoss(),tol=1e-5,verbose=True):
    
#     past_loss = 1e10
#     for epoch_no in range(num_epochs):
         
#         loss_dict,_ = run_epoch_batch(epoch_no,model,X,edge_indices,Y,model_type,optimizer,
#                                       node_indices=node_indices,edge_attr=edge_attr,device=device,
#                                       train=True,pred_criterion=pred_criterion,
#                                       verbose=verbose,
#                                       intervention_loss=intervention_loss,
#                                       lam_causal=lam_causal,
#                                       n_interventions_per_node=n_interventions_per_node)

#         current_loss = loss_dict['total_loss']
#         if early_stop and abs(past_loss - current_loss)/abs(past_loss) < tol:
#             break
            
#         past_loss = current_loss
        
def train_model_dataloader(model,dataloader,model_type,optimizer,device=0,
                           num_epochs=10,pred_criterion=nn.BCELoss(),
                           early_stop=True,tol=1e-3,verbose=True,
                           intervention_loss=False,lam_causal=1,
                           n_layers=1,
                           n_interventions_per_node=10,task='npp',
                           valid_dataloader=None):
        
    past_loss = 1e10
    loss_df = {}
    for epoch_no in range(num_epochs):
        
        start = time.time()
        loss_dict,_ = run_epoch_dataloader(epoch_no,model,dataloader,model_type=model_type,
                                           optimizer=optimizer,device=device,
                                           verbose=verbose,train=True,
                                           pred_criterion=pred_criterion,
                                           n_layers=n_layers,
                                           intervention_loss=intervention_loss,
                                           lam_causal=lam_causal,
                                           n_interventions_per_node=n_interventions_per_node,
                                           task=task,mask_attr='train_mask')
        if epoch_no == 0:
            loss_df = {k: [] for k in loss_dict.keys()}
            loss_df['epoch_no'] = []
            if early_stop:
                loss_df['val_pred_loss'] = []
        
        for k,v in loss_dict.items():
            loss_df[k].append(v)
        loss_df['epoch_no'].append(epoch_no)
          
        if 'early_stop100' in model_type:
            N = 100
        elif 'early_stop10' in model_type:
            N = 10
        elif 'early_stop20' in model_type:
            N = 20
        elif 'early_stop50' in model_type:
            N = 50
        elif 'early_stop3' in model_type:
            N = 3
        elif 'early_stop1' in model_type:
            N = 1
        else:
            N = 5
            
        if early_stop:
            val_loss_dict,_ = run_epoch_dataloader(epoch_no,model,valid_dataloader,model_type=model_type,
                                           optimizer=optimizer,device=device,
                                           verbose=verbose,train=False,
                                           pred_criterion=pred_criterion,
                                           intervention_loss=intervention_loss,
                                           task=task,mask_attr='val_mask')
            loss_df['val_pred_loss'].append(val_loss_dict['pred_loss'])
            
            if epoch_no >= 2*N-1:
                prev_last_N = np.array(loss_df['val_pred_loss'][-2*N:-N]).mean()
                last_N = np.array(loss_df['val_pred_loss'][-N:]).mean()
                if (prev_last_N - last_N)/abs(prev_last_N) < tol:
                    print('--- EARLY STOP: EPOCH {} ---'.format(epoch_no))
                    break
                
        end = time.time()
                
        if verbose and (epoch_no+1) % 5 == 0:
            # mode = 'train' if train else 'test'
            print_str = 'Epoch {} ({:.5f} seconds)'.format(
                epoch_no,end-start)
            print_str += ': total loss {:.5f}, pred loss {:.5f}'.format(loss_df['total_loss'][-1],
                                                                        loss_df['pred_loss'][-1])
            if intervention_loss:
                print_str += ', interv loss {:.5f}'.format(loss_df['intervention_loss'][-1])
            if early_stop:
                print_str += ', val pred loss {:.5f}'.format(loss_df['val_pred_loss'][-1])
            print(print_str)
        
    return pd.DataFrame(loss_df)
                    
def run_epoch_dataloader(epoch_no,model,dataloader,model_type='causal',
                    optimizer=None,device=0,
                    verbose=True,train=True,
                    pred_criterion=nn.BCELoss(),
                    intervention_loss=False,n_layers=1,
                    lam_causal=1,n_interventions_per_node=10,
                    task='npp',mask_attr=None):
        
    np.random.seed(epoch_no)
    torch.manual_seed(epoch_no)
    
    start = time.time()
    n_examples = 0
    total_loss = 0
    total_pred_loss = 0
    total_intervention_loss = 0
    
    model = model.to(device)
    pred_criterion = pred_criterion.to(device)

    preds_list = []
    for batch in dataloader:
                
        batch = batch.to(device)
        X = batch.x
        batch_edge_index = batch.edge_index
        batch_edge_attr = batch.edge_attr if hasattr(batch, 'edge_attr') else None

        if task == 'npp' or task == 'gpp':
            Y = batch.y
            batch_edge_index_pred  = None
        elif task == 'lpp':
            is_undirected = True
            if train:
                
                neg_edge_index = negative_sampling(batch_edge_index).to(batch.edge_index.device)
                Y = torch.cat([torch.ones(batch_edge_index.size(1)),
                               torch.zeros(batch_edge_index.size(1))])
                batch_edge_index_pred = torch.cat([batch_edge_index,
                                                   neg_edge_index],dim=1)
                Y = Y.to(batch.edge_index.device)
                batch_edge_index_pred = batch_edge_index_pred.to(batch.edge_index.device)

            else:
                Y = batch.edge_label
                batch_edge_index_pred = torch.clone(batch.edge_label_index)

        # node-level predictions
        if task == 'npp':
            batch_ptr = None
            preds,(batch_edge_index,attn_weights) = model(X,batch_edge_index,batch_edge_attr)
        elif task == 'gpp':
            batch_ptr = batch.ptr #.to(device)
            preds,(batch_edge_index,attn_weights) = model(X,batch_edge_index,batch_ptr,batch_edge_attr)
        elif task == 'lpp':
            batch_ptr = None
            # print(dataloader.data.x.shape,dataloader.data.edge_index.max()) #,batch.edge_label_index.max())
            preds,(_,attn_weights) = model(X,batch_edge_index,batch_edge_index_pred,
                                                          batch_edge_attr)
            
        batch_mask = getattr(batch,mask_attr) if hasattr(batch,str(mask_attr)) \
                                else torch.ones(Y.size(0)).bool().to(Y.device)
        nan_mask = ~torch.isnan(Y[batch_mask])

        if 'weight' in model_type:
            percent_pos = torch.nanmean(batch.y,0)
            weight = (1-percent_pos)/percent_pos
            weight = torch.tile(weight,(batch.y.size(0),1))
            weight = torch.nan_to_num(weight,posinf=1)
            weight[(batch.y == 0) | (batch.y.isnan())] = 1
            pred_criterion.weight = weight[nan_mask]

        # prediction loss
        batch_loss = 0
        pred_loss = pred_criterion(preds[batch_mask][nan_mask],
                                   Y[batch_mask][nan_mask]).mean()

        if train:
            
            if 'adversarial' in model_type:
                sampling = 'adversarial'
                edge_sampling_values = torch.stack([attn.mean(1) for attn 
                                                   in attn_weights]).mean(0)
            else:
                sampling = 'uniform'
                edge_sampling_values = None
                # attn_weights.retain_grad()
                
            weight_by_degree = 'wbd' in model_type
                
            # causal intervention loss
            causal_interv_loss = torch.zeros(1)
            loss_ratio = 'ratio' in model_type
            loss_ratio = 'thresh' if 'thresh' in model_type else loss_ratio
            use_labelprop = 'labelprop' in model_type
            
            if intervention_loss:
                
                if task == 'npp':
                    node_indices = torch.arange(batch_edge_index.max()+1)[batch_mask]
                elif task == 'gpp':
                    node_indices = torch.arange(Y.size(0))
                elif task == 'lpp':
                    node_indices = torch.arange(batch_edge_index.max()+1)
                    
                shuffle_effect = 'shuffle' in model_type
                
                pred_criterion.weight = None
                causal_interv_loss = compute_intervention_loss(
                                        model,X,node_indices,
                                        batch_edge_index,Y,preds,attn_weights,
                                        device=device,edge_attr=batch_edge_attr,
                                        pred_criterion=pred_criterion,sampling=sampling,
                                        edge_sampling_values=edge_sampling_values,
                                        weight_by_degree=weight_by_degree,
                                        n_layers=n_layers,
                                        n_interventions_per_node=n_interventions_per_node,
                                        task=task,ptr=batch_ptr,
                                        edge_indices_pred=batch_edge_index_pred,
                                        loss_ratio=loss_ratio,shuffle_effect=shuffle_effect,
                                        use_labelprop=use_labelprop,lp_batch=batch)
            
            if isinstance(lam_causal,list):
                lam_causal_tensor = torch.tensor(lam_causal).float().to(causal_interv_loss.device)
                causal_loss = (lam_causal_tensor*causal_interv_loss).sum()
            else:
                causal_loss = (lam_causal*causal_interv_loss).sum()
            
            batch_loss = pred_loss + causal_loss
            
            optimizer.zero_grad()
            batch_loss.backward()
                        
            optimizer.step()
            
            # if 'adversarial' in model_type:
            #     attn_grads = attn_weights.grad.mean(1)
            
            batch_size = preds[batch_mask].size(0)
            n_examples += batch_size
            
            total_loss += batch_loss.detach().cpu().data.numpy()*batch_size
            total_pred_loss += pred_loss.detach().cpu().data.numpy()*batch_size
            total_intervention_loss += causal_loss.detach().cpu().data.numpy()*batch_size
            
        else:
            batch_size = preds[batch_mask].size(0)
            n_examples += batch_size
            total_pred_loss += pred_loss.detach().cpu().data.numpy()*batch_size
            preds_list.append(preds[batch_mask].detach())
                    
    end = time.time()
        
    loss_dict = {'total_loss': total_loss/n_examples,
                 'pred_loss': total_pred_loss/n_examples,
                 'intervention_loss': total_intervention_loss/n_examples
                 }
    preds = None if train else torch.cat(preds_list)
    
    return loss_dict,preds