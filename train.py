import numpy as np
import time

import torch
import torch.nn as nn
from torch_geometric.utils import negative_sampling
from torch_geometric.transforms import RandomLinkSplit

from torch_geometric.loader import NeighborSampler

from utils import *
from causal import *
        
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
            loss_ratio = 'deg_wt' if 'deg_wt' in model_type else loss_ratio

            use_labelprop = 'labelprop' in model_type
            
            if intervention_loss:
                
                if task == 'npp':
                    node_indices = torch.arange(batch_edge_index.max()+1)[batch_mask]
                elif task == 'gpp':
                    node_indices = torch.arange(Y.size(0))
                elif task == 'lpp':
                    node_indices = torch.arange(batch_edge_index.max()+1)
                    
                shuffle_effect = 'shuffle' in model_type
                shuffle_effect = 'unif' if 'unif' in model_type else shuffle_effect
                
                if task == 'gpp':
                    pred_criterion.weight = None # for weighting by % positives
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