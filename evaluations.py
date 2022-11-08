import numpy as np
import os
from matplotlib import pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

from scipy.stats import kendalltau,rankdata
from scipy.stats import wilcoxon
from utils import *
from train import *
from run import instantiate_model
from torch_geometric.utils import degree

from collections import Counter
from sklearn.metrics import dcg_score
from scipy.special import kl_div

def evaluate_single_model(model_type,evaluator,model,dataloader,device=0,task='gpp',mask_attr=None):
    
    # _,preds = run_epoch_dataloader(0,model,dataloader,model_type=model_type,
    #                              device=device,verbose=False,train=False,task=task,
    #                              mask_attr=mask_attr)
    
    model = model.to(device)
    if task != 'lpp':
        y_pred_list = []
        y_true_list = []
        for batch in dataloader:
            batch = batch.to(device)
            batch_mask = getattr(batch,mask_attr) if hasattr(batch,str(mask_attr)) \
                                    else torch.ones(batch.y.size(0)).bool().to(batch.y.device)
            if task == 'gpp':
                batch_pred,_ = model(batch.x,batch.edge_index,batch.ptr,batch.edge_attr)
            else:
                batch_pred,_ = model(batch.x,batch.edge_index,batch.edge_attr)
            y_pred_list.append(batch_pred[batch_mask])
            y_true_list.append(batch.y[batch_mask])
            
        y_pred = torch.cat(y_pred_list)
        if task == 'npp':
            y_pred = y_pred.argmax(1).unsqueeze(-1)
        else:
            y_pred = y_pred
        y_pred = y_pred.cpu().data.numpy()
        
        y_true = torch.cat(y_true_list)
        y_true = y_true.unsqueeze(-1) if y_true.dim() == 1 else y_true
        y_true = y_true.cpu().data.numpy()

        result_dict = evaluator.eval({"y_true": y_true,"y_pred": y_pred})

    elif task == 'lpp':
        y_pred_pos = []
        y_pred_neg = []
        
        for batch in dataloader:
            batch = batch.to(device)
            batch_preds,_ = model(batch.x,batch.edge_index,batch.edge_label_index,
                                       batch.edge_attr)
            y_pred_pos.append(batch_preds[batch.edge_label.nonzero().squeeze()])
            y_pred_neg.append(batch_preds[(batch.edge_label == 0).nonzero().squeeze()])
        y_pred_pos = torch.cat(y_pred_pos).cpu()
        y_pred_neg = torch.cat(y_pred_neg).cpu()

        result_dict = evaluator.eval({"y_pred_pos": y_pred_pos,"y_pred_neg": y_pred_neg})

    torch.cuda.empty_cache() 

    return result_dict

def evaluate_single_model_loss(model_type,pred_criterion,model,dataloader,device=0,task='gpp',mask_attr=None):

    model = model.to(device)
    if task != 'lpp':
        y_pred_list = []
        y_true_list = []
        for batch in dataloader:
            batch = batch.to(device)
            batch_mask = getattr(batch,mask_attr) if hasattr(batch,str(mask_attr)) \
                                    else torch.ones(batch.y.size(0)).bool().to(batch.y.device)
            if task == 'gpp':
                batch_pred,_ = model(batch.x,batch.edge_index,batch.ptr,batch.edge_attr)
            else:
                batch_pred,_ = model(batch.x,batch.edge_index,batch.edge_attr)
            y_pred_list.append(batch_pred[batch_mask])
            y_true_list.append(batch.y[batch_mask])
            
        y_pred = torch.cat(y_pred_list)
        
        y_true = torch.cat(y_true_list)

        loss = pred_criterion(y_pred,y_true).mean().cpu().data.numpy()
        
#     elif task == 'lpp':
#         y_pred_pos = []
#         y_pred_neg = []
        
#         for batch in dataloader:
#             batch = batch.to(device)
#             batch_preds,_ = model(batch.x,batch.edge_index,batch.edge_label_index,
#                                        batch.edge_attr)
#             y_pred_pos.append(batch_preds[batch.edge_label.nonzero().squeeze()])
#             y_pred_neg.append(batch_preds[(batch.edge_label == 0).nonzero().squeeze()])
#         y_pred_pos = torch.cat(y_pred_pos).cpu()
#         y_pred_neg = torch.cat(y_pred_neg).cpu()

#         result_dict = evaluator.eval({"y_pred_pos": y_pred_pos,"y_pred_neg": y_pred_neg})

    torch.cuda.empty_cache() 

    return loss

from sklearn.metrics import average_precision_score,roc_auc_score

def label_agreement(model,dataloader,task,device=0,weight_by_degree=False):
    
    model = model.to(device)
    
    y_true = []
    y_pred = []
    for batch in dataloader:
        batch = batch.to(device)
        if task == 'gpp':
            batch_pred,(batch_edge_index,batch_attn_weights_list) = model(batch.x,batch.edge_index,batch.ptr,batch.edge_attr)
        else:
            batch_pred,(batch_edge_index,batch_attn_weights_list) = model(batch.x,batch.edge_index,batch.edge_attr)

        attn_weights = torch.cat(batch_attn_weights_list,1).mean(1)
        label1,label2 = batch.y[batch.edge_index]
        
        if weight_by_degree:
            deg = degree(batch.edge_index[1],batch.x.size(0))
            attn_weights *= deg[batch.edge_index[1]]
            
        y_true.extend((label1 == label2).float().cpu().data.numpy())
        y_pred.extend(attn_weights.cpu().data.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    torch.cuda.empty_cache() 
    
    return average_precision_score(y_true,y_pred),roc_auc_score(y_true,y_pred),y_true.mean()

def label_agreement_kl(model,dataloader,task,device=0,weight_by_degree=False):
    
    model = model.to(device)
    
    y_true = []
    y_pred = []
    kl_list = []
    for batch in dataloader:
        batch = batch.to(device)
        if task == 'gpp':
            batch_pred,(batch_edge_index,batch_attn_weights_list) = model(batch.x,batch.edge_index,batch.ptr,batch.edge_attr)
        else:
            batch_pred,(batch_edge_index,batch_attn_weights_list) = model(batch.x,batch.edge_index,batch.edge_attr)

        attn_weights = torch.cat(batch_attn_weights_list,1).mean(1)
        label1,label2 = batch.y[batch.edge_index]
        
        agree = (label1 == label2).float().cpu().data.numpy()
        
        
        edge_index = batch_edge_index.cpu().data.numpy()
        for target_idx in list(set(edge_index[1])):
            attn_weights_neigh = attn_weights[edge_index[1] == target_idx]
            
            agree_neigh = attn_weights[edge_index[1] == target_idx]
            agree_neigh *= 1./agree_neigh.sum()
            
            attn_weights_neigh = attn_weights_neigh.cpu().data.numpy()
            agree_neigh = agree_neigh.cpu().data.numpy()
            
            kl_list.extend(kl_div(attn_weights_neigh,agree_neigh))

    torch.cuda.empty_cache() 
    
    return np.mean(kl_list)



def attn_freq_dcg(model,dataloader,task,device=0,weight_by_degree=False):
    
    model = model.to(device)
    
    pairs_list = []
    attn_weights_list = []
    for batch in dataloader:
        batch = batch.to(device)
        if task == 'gpp':
            batch_pred,(batch_edge_index,batch_attn_weights_list) = model(batch.x,batch.edge_index,batch.ptr,batch.edge_attr)
        else:
            batch_pred,(batch_edge_index,batch_attn_weights_list) = model(batch.x,batch.edge_index,batch.edge_attr)

        attn_weights = torch.cat(batch_attn_weights_list,1).mean(1)
        
        if weight_by_degree:
            deg = degree(batch.edge_index[1],batch.x.size(0))
            attn_weights *= deg[batch.edge_index[1]]
        
        attn_weights_list.extend(attn_weights.cpu().data.numpy())
        
        label1,label2 = batch.y[batch.edge_index].cpu().data.numpy()
        pairs = [(i,j) for i,j in zip(label1,label2)]
        pairs_list.extend(pairs)
        
    counts = Counter(pairs_list)
    in_degrees = Counter([j for i,j in pairs_list])

    data_dict = {k: [] for k in ['i','j','freq']}
    for (i,j),v in counts.items():
        data_dict['i'].append(i)
        data_dict['j'].append(j)
        data_dict['freq'].append(v/in_degrees[j])
    freq_df = pd.DataFrame(data_dict)
    freq_df.index = [(i,j) for i,j in freq_df[['i','j']].values]

    attn_df = pd.DataFrame()
    attn_df['i'] = np.array(pairs_list)[:,0]
    attn_df['j'] = np.array(pairs_list)[:,1]
    attn_df['attn'] = attn_weights_list

    attn_df = attn_df.groupby(['i','j']).mean().reset_index()
    attn_df.index = [(i,j) for i,j in attn_df[['i','j']].values]
    freq_df = freq_df.loc[attn_df.index.values]
    
    torch.cuda.empty_cache() 
    
    return dcg_score([freq_df['freq'].values],[attn_df['attn'].values])

class KendallEvaluator:
    def __init__(dataset_name):
        pass

    def eval(self,data_dict):

        rank_y_pred = rankdata(data_dict["y_pred"])
        rank_y_true = rankdata(data_dict["y_true"])

        return {'kendall': kendalltau(rank_y_pred,rank_y_true)[0]}

def load_model(dataset_name,model_type,heads,dim_in,dim_hidden,dim_out,n_layers,edge_dim,save_dir,
               base=True,lc=None,ni=None,n_embeddings=None):
    
    model = instantiate_model(dataset_name,model_type,dim_in,dim_hidden,dim_out,
                      heads,n_layers,edge_dim,n_embeddings=n_embeddings)

    if base:
        model_file_name = '{}.base.{}heads.{}hd.nl{}.pt'.format(model_type,heads,dim_hidden,n_layers)
    else:
        model_file_name = '{}.{}heads.{}hd.nl{}.lc{}.ni{}.pt'.format(model_type,heads,dim_hidden,n_layers,lc,ni)

    model.load_state_dict(torch.load(os.path.join(save_dir,model_file_name)))

    return model

def load_model_gcn(dataset_name,model_type,heads,dim_in,dim_hidden,dim_out,n_layers,edge_dim,save_dir,
               base=True,lc=None,ni=None,n_embeddings=None,attn_thresh=None,base_gcn=False,verbose=False,
               gcn_dim_hidden=200):
    
    model = instantiate_model(dataset_name,'gcnconv',dim_in,gcn_dim_hidden,dim_out,
                      heads,n_layers,edge_dim,n_embeddings=n_embeddings)

    if base:
        model_file_name = '{}.base.{}heads.{}hd.nl{}.gcn.thresh{}.pt'.format(model_type,heads,dim_hidden,n_layers,attn_thresh)
    elif base_gcn:
        model_file_name = '{}.{}hd.nl{}.gcn.base.pt'.format(model_type,dim_hidden,n_layers)
    else:
        model_file_name = '{}.{}heads.{}hd.nl{}.lc{}.ni{}.gcn.thresh{}.pt'.format(model_type,heads,dim_hidden,n_layers,
                                                                              lc,ni,attn_thresh)
        
    if verbose:
        print(model_file_name)
        
    model.load_state_dict(torch.load(os.path.join(save_dir,model_file_name)))

    return model

def add_results_to_dict(data_dict,model_type,base,n_layers,heads,dim_hidden,lc,ni,eval_set,eval_metric,results):
    
    data_dict['model'].append(model_type)
    data_dict['base'].append(base)
    data_dict['n_layers'].append(n_layers)
    data_dict['heads'].append(heads)
    data_dict['dim_hidden'].append(dim_hidden)
    data_dict['lc'].append(lc)
    data_dict['ni'].append(ni)
    data_dict['set'].append(eval_set)
    data_dict[eval_metric].append(results)

    return data_dict

def evaluate_models(dataset_name,valid_loader,test_loader,evaluator,save_dir,params_dict,
                    eval_metric='acc',device=0,suffix='interv',task='npp'):
    
    data_dict = {k: [] for k in ['model','base','n_layers','dim_hidden',
                               'heads','lc','ni','set',eval_metric]}
    
    for model_base in params_dict['model']:
        model_type = '{}.{}'.format(model_base,suffix)
    
        for dim_hidden in params_dict['hd']:
            for n_layers in params_dict['nl']:
                for heads in params_dict['heads']:
                    dim_in,dim_out,edge_dim,_ = get_dataset_params(dataset_name,valid_loader,dim_hidden)
                    n_embeddings = valid_loader.data.num_nodes if dataset_name == 'ogbl-ddi' else None

                    model = load_model(dataset_name,model_type,heads,dim_in,dim_hidden,dim_out,
                                         n_layers,edge_dim,save_dir,base=True,lc=None,ni=None,n_embeddings=n_embeddings)
                    valid_results = evaluate_single_model(model_type,evaluator,model,valid_loader,task=task,device=device,mask_attr='val_mask')[eval_metric]
                    test_results = evaluate_single_model(model_type,evaluator,model,test_loader,task=task,device=device,mask_attr='test_mask')[eval_metric]

                    add_results_to_dict(data_dict,model_type,True,n_layers,heads,dim_hidden,0,0,'valid',eval_metric,valid_results)
                    add_results_to_dict(data_dict,model_type,True,n_layers,heads,dim_hidden,0,0,'test',eval_metric,test_results)
                    
                    for lc in params_dict['lc']:
                        for ni in params_dict['ni']:
                            model = load_model(dataset_name,model_type,heads,dim_in,dim_hidden,dim_out,
                                                 n_layers,edge_dim,save_dir,base=False,lc=lc,ni=ni,n_embeddings=n_embeddings)
                            valid_results = evaluate_single_model(model_type,evaluator,model,valid_loader,task=task,device=device,mask_attr='val_mask')[eval_metric]
                            test_results = evaluate_single_model(model_type,evaluator,model,test_loader,task=task,device=device,mask_attr='test_mask')[eval_metric]

                            add_results_to_dict(data_dict,model_type,False,n_layers,heads,dim_hidden,lc,ni,'valid',eval_metric,valid_results)
                            add_results_to_dict(data_dict,model_type,False,n_layers,heads,dim_hidden,lc,ni,'test',eval_metric,test_results)
                            
    results_df = pd.DataFrame(data_dict)
    results_df['params'] = ['nl{}.hd{}.{}heads.lc{}.ni{}'.format(int(n_layers),int(dim_hidden),int(heads),lc,int(ni)) 
                        for n_layers,dim_hidden,heads,lc,ni in results_df[['n_layers','dim_hidden','heads','lc','ni']].values]
    
    return results_df

def evaluate_models_loss(dataset_name,valid_loader,test_loader,pred_criterion,save_dir,params_dict,
                         device=0,suffix='interv',task='npp'):
    
    data_dict = {k: [] for k in ['model','base','n_layers','dim_hidden',
                               'heads','lc','ni','set','loss']}
    
    for model_base in params_dict['model']:
        model_type = '{}.{}'.format(model_base,suffix)
    
        for dim_hidden in params_dict['hd']:
            for n_layers in params_dict['nl']:
                for heads in params_dict['heads']:
                    dim_in,dim_out,edge_dim,_ = get_dataset_params(dataset_name,valid_loader,dim_hidden)
                    n_embeddings = valid_loader.data.num_nodes if dataset_name == 'ogbl-ddi' else None

                    model = load_model(dataset_name,model_type,heads,dim_in,dim_hidden,dim_out,
                                         n_layers,edge_dim,save_dir,base=True,lc=None,ni=None,n_embeddings=n_embeddings)
                    valid_results = evaluate_single_model_loss(model_type,pred_criterion,model,valid_loader,task=task,device=device,mask_attr='val_mask')
                    test_results = evaluate_single_model_loss(model_type,pred_criterion,model,test_loader,task=task,device=device,mask_attr='test_mask')

                    add_results_to_dict(data_dict,model_type,True,n_layers,heads,dim_hidden,0,0,'valid','loss',valid_results)
                    add_results_to_dict(data_dict,model_type,True,n_layers,heads,dim_hidden,0,0,'test','loss',test_results)
                    
                    for lc in params_dict['lc']:
                        for ni in params_dict['ni']:
                            model = load_model(dataset_name,model_type,heads,dim_in,dim_hidden,dim_out,
                                                 n_layers,edge_dim,save_dir,base=False,lc=lc,ni=ni,n_embeddings=n_embeddings)
                            valid_results = evaluate_single_model_loss(model_type,pred_criterion,model,valid_loader,task=task,device=device,mask_attr='val_mask')
                            test_results = evaluate_single_model_loss(model_type,pred_criterion,model,test_loader,task=task,device=device,mask_attr='test_mask')

                            add_results_to_dict(data_dict,model_type,False,n_layers,heads,dim_hidden,lc,ni,'valid','loss',valid_results)
                            add_results_to_dict(data_dict,model_type,False,n_layers,heads,dim_hidden,lc,ni,'test','loss',test_results)
                            
    results_df = pd.DataFrame(data_dict)
    results_df['params'] = ['nl{}.hd{}.{}heads.lc{}.ni{}'.format(int(n_layers),int(dim_hidden),int(heads),lc,int(ni)) 
                        for n_layers,dim_hidden,heads,lc,ni in results_df[['n_layers','dim_hidden','heads','lc','ni']].values]
    
    return results_df

def evaluate_label_agreement(dataset_name,dataloader,save_dir,params_dict,
                    device=0,suffix='interv',task='npp',mask_attr='test_mask',
                    weight_by_degree=False):
    
    data_dict = {k: [] for k in ['model','base','n_layers','dim_hidden',
                               'heads','lc','ni','set','auroc','ap','pos_rate']}
    
    for model_base in params_dict['model']:
        model_type = '{}.{}'.format(model_base,suffix)
        for dim_hidden in params_dict['hd']:
            for n_layers in params_dict['nl']:
                for heads in params_dict['heads']:
                    dim_in,dim_out,edge_dim,_ = get_dataset_params(dataset_name,dataloader,dim_hidden)
                    n_embeddings = dataloader.data.num_nodes if dataset_name == 'ogbl-ddi' else None

                    model = load_model(dataset_name,model_type,heads,dim_in,dim_hidden,dim_out,
                                         n_layers,edge_dim,save_dir,base=True,lc=None,ni=None,n_embeddings=n_embeddings)
                    ap,auroc,pos_rate = label_agreement(model,dataloader,task=task,device=device,weight_by_degree=weight_by_degree)

                    add_results_to_dict(data_dict,model_type,True,n_layers,heads,dim_hidden,0,0,'test','ap',ap)
                    data_dict['auroc'].append(auroc)
                    data_dict['pos_rate'].append(pos_rate)
                    
                    for lc in params_dict['lc']:
                        for ni in params_dict['ni']:
                            model = load_model(dataset_name,model_type,heads,dim_in,dim_hidden,dim_out,
                                                 n_layers,edge_dim,save_dir,base=False,lc=lc,ni=ni,n_embeddings=n_embeddings)
                            ap,auroc,pos_rate = label_agreement(model,dataloader,task=task,device=device,weight_by_degree=weight_by_degree)
                            add_results_to_dict(data_dict,model_type,False,n_layers,heads,dim_hidden,lc,ni,'test','ap',ap)
                            data_dict['auroc'].append(auroc)
                            data_dict['pos_rate'].append(pos_rate)
                            
    results_df = pd.DataFrame(data_dict)
    results_df['params'] = ['nl{}.hd{}.{}heads.lc{}.ni{}'.format(int(n_layers),int(dim_hidden),int(heads),lc,int(ni)) 
                        for n_layers,dim_hidden,heads,lc,ni in results_df[['n_layers','dim_hidden','heads','lc','ni']].values]
    
    return results_df

def evaluate_label_agreement_kl(dataset_name,dataloader,save_dir,params_dict,
                    device=0,suffix='interv',task='npp',mask_attr='test_mask',
                    weight_by_degree=False):
    
    data_dict = {k: [] for k in ['model','base','n_layers','dim_hidden',
                               'heads','lc','ni','set','kl']}
    
    for model_base in params_dict['model']:
        model_type = '{}.{}'.format(model_base,suffix)
        for dim_hidden in params_dict['hd']:
            for n_layers in params_dict['nl']:
                for heads in params_dict['heads']:
                    dim_in,dim_out,edge_dim,_ = get_dataset_params(dataset_name,dataloader,dim_hidden)
                    n_embeddings = dataloader.data.num_nodes if dataset_name == 'ogbl-ddi' else None

                    model = load_model(dataset_name,model_type,heads,dim_in,dim_hidden,dim_out,
                                         n_layers,edge_dim,save_dir,base=True,lc=None,ni=None,n_embeddings=n_embeddings)
                    mean_kl = label_agreement_kl(model,dataloader,task=task,device=device,weight_by_degree=weight_by_degree)
                    add_results_to_dict(data_dict,model_type,True,n_layers,heads,dim_hidden,0,0,'test','kl',mean_kl)

                    for lc in params_dict['lc']:
                        for ni in params_dict['ni']:
                            model = load_model(dataset_name,model_type,heads,dim_in,dim_hidden,dim_out,
                                                 n_layers,edge_dim,save_dir,base=False,lc=lc,ni=ni,n_embeddings=n_embeddings)
                            mean_kl = label_agreement_kl(model,dataloader,task=task,device=device,weight_by_degree=weight_by_degree)
                            add_results_to_dict(data_dict,model_type,False,n_layers,heads,dim_hidden,lc,ni,'test','kl',mean_kl)
                            
    results_df = pd.DataFrame(data_dict)
    results_df['params'] = ['nl{}.hd{}.{}heads.lc{}.ni{}'.format(int(n_layers),int(dim_hidden),int(heads),lc,int(ni)) 
                        for n_layers,dim_hidden,heads,lc,ni in results_df[['n_layers','dim_hidden','heads','lc','ni']].values]
    
    return results_df

def evaluate_attn_freq_dcg(dataset_name,dataloader,save_dir,params_dict,
                    device=0,suffix='interv',task='npp',mask_attr='test_mask',
                    weight_by_degree=False):
    
    data_dict = {k: [] for k in ['model','base','n_layers','dim_hidden',
                               'heads','lc','ni','set','dcg']}
    
    for model_base in params_dict['model']:
        model_type = '{}.{}'.format(model_base,suffix)
        for dim_hidden in params_dict['hd']:
            for n_layers in params_dict['nl']:
                for heads in params_dict['heads']:
                    dim_in,dim_out,edge_dim,_ = get_dataset_params(dataset_name,dataloader,dim_hidden)
                    n_embeddings = dataloader.data.num_nodes if dataset_name == 'ogbl-ddi' else None

                    model = load_model(dataset_name,model_type,heads,dim_in,dim_hidden,dim_out,
                                         n_layers,edge_dim,save_dir,base=True,lc=None,ni=None,n_embeddings=n_embeddings)
                    dcg = attn_freq_dcg(model,dataloader,task=task,device=device,weight_by_degree=weight_by_degree)
                    add_results_to_dict(data_dict,model_type,True,n_layers,heads,dim_hidden,0,0,'test','dcg',dcg)

                    for lc in params_dict['lc']:
                        for ni in params_dict['ni']:
                            model = load_model(dataset_name,model_type,heads,dim_in,dim_hidden,dim_out,
                                                 n_layers,edge_dim,save_dir,base=False,lc=lc,ni=ni,n_embeddings=n_embeddings)
                            dcg = attn_freq_dcg(model,dataloader,task=task,device=device,weight_by_degree=weight_by_degree)
                            add_results_to_dict(data_dict,model_type,False,n_layers,heads,dim_hidden,lc,ni,'test','dcg',dcg)

    results_df = pd.DataFrame(data_dict)
    results_df['params'] = ['nl{}.hd{}.{}heads.lc{}.ni{}'.format(int(n_layers),int(dim_hidden),int(heads),lc,int(ni)) 
                        for n_layers,dim_hidden,heads,lc,ni in results_df[['n_layers','dim_hidden','heads','lc','ni']].values]
    
    return results_df

def evaluate_models_gcn_base(dataset_name,evaluator,save_dir,params_dict,
                    eval_metric='acc',device=0,suffix='interv',task='npp',batch_size=5000,attn_thresh=0.1,trial_no=0,
                    gcn_dim_hidden=20,split_no=0):
        
    orig_save_dir = os.path.join(save_dir,'models')
    rewire_save_dir = os.path.join(save_dir,'models_rewire')
    
    data_dict = {k: [] for k in ['model','base','n_layers','dim_hidden',
                               'heads','lc','ni','set',eval_metric]}
    
    for model_base in params_dict['model']:
        model_type = '{}.{}'.format(model_base,suffix)
    
        for dim_hidden in params_dict['hd']:
            for n_layers in params_dict['nl']:
                  
                # reload dataloader
                train_loader,valid_loader,test_loader = load_dataloader(dataset_name,batch_size=batch_size,shuffle_train=False,split_no=split_no)
                dim_in,dim_out,edge_dim,_ = get_dataset_params(dataset_name,valid_loader,dim_hidden)
                n_embeddings = valid_loader.data.num_nodes if dataset_name == 'ogbl-ddi' else None

                model_gcn = load_model_gcn(dataset_name,model_type + '.trial{}'.format(trial_no),0,dim_in,dim_hidden,dim_out,
                                         n_layers,edge_dim,rewire_save_dir,base=False,lc=None,ni=None,n_embeddings=n_embeddings,
                                         attn_thresh=attn_thresh,base_gcn=True,gcn_dim_hidden=gcn_dim_hidden,verbose=False)

                train_results = evaluate_single_model(model_type,evaluator,model_gcn,train_loader,
                                                        task=task,device=device,mask_attr='train_mask')[eval_metric]
                valid_results = evaluate_single_model(model_type,evaluator,model_gcn,valid_loader,
                                                        task=task,device=device,mask_attr='val_mask')[eval_metric]
                test_results = evaluate_single_model(model_type,evaluator,model_gcn,test_loader,
                                                       task=task,device=device,mask_attr='test_mask')[eval_metric]
                
                add_results_to_dict(data_dict,model_type,True,n_layers,0,dim_hidden,0,0,'train',eval_metric,train_results)
                add_results_to_dict(data_dict,model_type,True,n_layers,0,dim_hidden,0,0,'valid',eval_metric,valid_results)
                add_results_to_dict(data_dict,model_type,True,n_layers,0,dim_hidden,0,0,'test',eval_metric,test_results)
                
    results_df = pd.DataFrame(data_dict)
    results_df['params'] = ['GCN.nl{}.hd{}.{}heads.lc{}.ni{}'.format(int(n_layers),int(dim_hidden),int(heads),lc,int(ni)) 
                        for n_layers,dim_hidden,heads,lc,ni in results_df[['n_layers','dim_hidden','heads','lc','ni']].values]
    
    return results_df

def evaluate_models_gcn(dataset_name,evaluator,save_dir,params_dict,
                    eval_metric='acc',device=0,suffix='interv',task='npp',batch_size=5000,attn_thresh=0.1,trial_no=0,
                    gcn_dim_hidden=20,split_no=0,weight_by_degree=False):
    
    orig_save_dir = os.path.join(save_dir,'models')
    rewire_save_dir = os.path.join(save_dir,'models_rewire')

    data_dict = {k: [] for k in ['model','base','n_layers','dim_hidden',
                               'heads','lc','ni','set',eval_metric]}
    for model_base in params_dict['model']:
        model_type = '{}.{}'.format(model_base,suffix)
    
        for dim_hidden in params_dict['hd']:
            for n_layers in params_dict['nl']:
                for heads in params_dict['heads']:
                    
                    # reload dataloader
                    train_loader,valid_loader,test_loader = load_dataloader(dataset_name,batch_size=batch_size,shuffle_train=False,split_no=split_no)
                    dim_in,dim_out,edge_dim,_ = get_dataset_params(dataset_name,valid_loader,dim_hidden)
                    n_embeddings = valid_loader.data.num_nodes if dataset_name == 'ogbl-ddi' else None

                    # graph attention network
                    model = load_model(dataset_name,model_type,heads,dim_in,dim_hidden,dim_out,
                                     n_layers,edge_dim,orig_save_dir,base=True,lc=None,ni=None,n_embeddings=n_embeddings)
                    rewired_train_loader = generate_rewired_dataloader(model,train_loader,attn_thresh=attn_thresh,
                                                                     batch_size=batch_size,shuffle=False,verbose=False,
                                                                     weight_by_degree=weight_by_degree)
                    rewired_valid_loader = generate_rewired_dataloader(model,valid_loader,attn_thresh=attn_thresh,
                                                                       batch_size=batch_size,shuffle=False,verbose=False,
                                                                       weight_by_degree=weight_by_degree)
                    rewired_test_loader = generate_rewired_dataloader(model,test_loader,attn_thresh=attn_thresh,
                                                                      batch_size=batch_size,shuffle=False,verbose=False,
                                                                      weight_by_degree=weight_by_degree)

                    model_gcn = load_model_gcn(dataset_name,model_type + '.trial{}'.format(trial_no),heads,dim_in,dim_hidden,dim_out,
                                         n_layers,edge_dim,rewire_save_dir,base_gcn=False,base=True,
                                         lc=None,ni=None,n_embeddings=n_embeddings,
                                         attn_thresh=attn_thresh,gcn_dim_hidden=gcn_dim_hidden,verbose=False)
                    
                    train_results = evaluate_single_model(model_type,evaluator,model_gcn,rewired_train_loader,
                                                        task=task,device=device,mask_attr='train_mask')[eval_metric]
                    valid_results = evaluate_single_model(model_type,evaluator,model_gcn,rewired_valid_loader,
                                                        task=task,device=device,mask_attr='val_mask')[eval_metric]
                    test_results = evaluate_single_model(model_type,evaluator,model_gcn,rewired_test_loader,
                                                       task=task,device=device,mask_attr='test_mask')[eval_metric]
                    
                    add_results_to_dict(data_dict,model_type,True,n_layers,heads,dim_hidden,0,0,'train',eval_metric,train_results)
                    add_results_to_dict(data_dict,model_type,True,n_layers,heads,dim_hidden,0,0,'valid',eval_metric,valid_results)
                    add_results_to_dict(data_dict,model_type,True,n_layers,heads,dim_hidden,0,0,'test',eval_metric,test_results)

                    for lc in params_dict['lc']:
                        for ni in params_dict['ni']:
                            # reload dataloader
                            train_loader,valid_loader,test_loader = load_dataloader(dataset_name,batch_size=batch_size,shuffle_train=False,split_no=split_no)

                            # graph attention network (causal)
                            model = load_model(dataset_name,model_type,heads,dim_in,dim_hidden,dim_out,
                                             n_layers,edge_dim,orig_save_dir,base=False,lc=lc,ni=ni,n_embeddings=n_embeddings)
                            rewired_train_loader = generate_rewired_dataloader(model,train_loader,attn_thresh=attn_thresh,
                                                                               batch_size=batch_size,shuffle=False,verbose=False,
                                                                               weight_by_degree=weight_by_degree)
                            rewired_valid_loader = generate_rewired_dataloader(model,valid_loader,attn_thresh=attn_thresh,
                                                                               batch_size=batch_size,shuffle=False,verbose=False,
                                                                               weight_by_degree=weight_by_degree)
                            rewired_test_loader = generate_rewired_dataloader(model,test_loader,attn_thresh=attn_thresh,
                                                                              batch_size=batch_size,shuffle=False,verbose=False,
                                                                              weight_by_degree=weight_by_degree)

                            model_gcn = load_model_gcn(dataset_name,model_type + '.trial{}'.format(trial_no),heads,dim_in,dim_hidden,dim_out,
                                                 n_layers,edge_dim,rewire_save_dir,base=False,lc=lc,ni=ni,n_embeddings=n_embeddings,
                                                 attn_thresh=attn_thresh,gcn_dim_hidden=gcn_dim_hidden,verbose=False)
                            
                            train_results = evaluate_single_model(model_type,evaluator,model_gcn,rewired_train_loader,
                                                                task=task,device=device,mask_attr='train_mask')[eval_metric]
                            valid_results = evaluate_single_model(model_type,evaluator,model_gcn,rewired_valid_loader,
                                                                task=task,device=device,mask_attr='val_mask')[eval_metric]
                            test_results = evaluate_single_model(model_type,evaluator,model_gcn,rewired_test_loader,
                                                               task=task,device=device,mask_attr='test_mask')[eval_metric]
                    
                            add_results_to_dict(data_dict,model_type,False,n_layers,heads,dim_hidden,lc,ni,'train',eval_metric,train_results)
                            add_results_to_dict(data_dict,model_type,False,n_layers,heads,dim_hidden,lc,ni,'valid',eval_metric,valid_results)
                            add_results_to_dict(data_dict,model_type,False,n_layers,heads,dim_hidden,lc,ni,'test',eval_metric,test_results)

    results_df = pd.DataFrame(data_dict)
    results_df['params'] = ['nl{}.hd{}.{}heads.lc{}.ni{}'.format(int(n_layers),int(dim_hidden),int(heads),lc,int(ni)) 
                        for n_layers,dim_hidden,heads,lc,ni in results_df[['n_layers','dim_hidden','heads','lc','ni']].values]

    return results_df

from matplotlib.ticker import FormatStrFormatter

model_color = {'gat': 'green', 'gatv2': 'blue', 'transformer': 'red'}

def plot_comparisons(results_df,eval_set,eval_metric='acc',alpha_feat='lc',save_path=None,plot=True,rotation=0,nbins=5):
    
    plot_dict = {k: [] for k in ['x','y','model','params']}
    results_df = results_df[results_df['set'] == eval_set]
    for model_base,group_df in results_df.groupby('model'):
        groupby_cols = ['n_layers','dim_hidden','heads']
        if 'split_no' in results_df.columns:
            groupby_cols.append('split_no')
            
        for params,param_df in group_df.groupby(groupby_cols):
            baseline = param_df[param_df['base']][eval_metric].values[0]

            param_df = param_df[~param_df['base']]
            plot_dict['x'].extend([baseline]*param_df.shape[0])
            plot_dict['y'].extend(param_df[eval_metric].values)
            plot_dict['params'].extend(param_df['params'].values)
            plot_dict['model'].extend([model_base]*param_df.shape[0])

    plot_df = pd.DataFrame(plot_dict)

    plot_df['model'] = [n.split('conv')[0] for n in plot_df['model']]
    plot_df['hd'] = [int(n.split('.')[1][2:]) for n in plot_df['params']]
    plot_df['lc'] = [float(n.split('lc')[1].split('.ni')[0]) for n in plot_df['params']]
    plot_df['ni'] = [int(n.split('ni')[1]) for n in plot_df['params']]
    plot_df['heads'] = [int(n.split('heads')[0].split('.')[-1]) for n in plot_df['params']]

    if plot:
        
        plt.figure(figsize=(4,4))
        
        min_value = plot_df[['x','y']].values.min()
        max_value = plot_df[['x','y']].values.max()
        min_value = min_value - 0.05*(max_value-min_value)
        max_value = max_value + 0.05*(max_value-min_value)
        
        plt.plot([min_value,max_value],[min_value,max_value],linestyle='--',linewidth=2,color='grey')
            
        for model,model_df in plot_df.groupby('model'):
            if alpha_feat is None:
                s = 150
            else:
                s = 150*(1+model_df[alpha_feat])/(1+model_df[alpha_feat]).max()

            plt.scatter(model_df['x'],model_df['y'], 
                      c=model_color[model],label=model,edgecolors='black',
                      s=s)
        
        plt.xlim([min_value,max_value])
        plt.ylim([min_value,max_value])

#         leg = plt.legend(fontsize=16,bbox_to_anchor=(1,1),frameon=False)
#         for lh in leg.legendHandles: 
#             lh.set_alpha(1)
#             lh._sizes = [100] 

        ticks = (np.linspace(min_value,max_value,20)*20).astype(int)/20
        ticks = sorted(list(set(ticks)))
        plt.xticks(fontsize=20,rotation=rotation)
        plt.yticks(fontsize=20)
        
        plt.locator_params(axis='x',nbins=nbins)
        plt.locator_params(axis='y',nbins=nbins)
        
        # plt.xlabel('Baseline\n{}'.format(eval_metric.upper()),fontsize=28)
        # plt.ylabel('Causal Attention\n{}'.format(eval_metric.upper()),fontsize=28)
        plt.xlabel(eval_metric.upper(),fontsize=24)
        plt.ylabel(eval_metric.upper(),fontsize=24)

        sns.despine()

        if save_path is not None:
            plt.savefig(save_path,dpi=500,bbox_inches='tight')
        plt.show()

    return plot_df

def plot_rand_comparison(plot_df_orig,plot_df_rand,eval_metric,alpha_feat='lc',save_path=None):
    
    plt.figure(figsize=(4,4))
    rand_diff = plot_df_rand['y']-plot_df_rand['x']
    causal_diff = plot_df_orig['y']-plot_df_orig['x']

    print(wilcoxon(rand_diff,causal_diff,alternative='less'))

    if alpha_feat is None:
        s = 50
    else:
        s = (1+plot_df_orig[alpha_feat])/(1+plot_df_orig[alpha_feat].max())*150
    
    plt.scatter(rand_diff,causal_diff,s=s,edgecolors='black')
    min_val = min(rand_diff.min(),causal_diff.min())*1.1
    max_val = max(rand_diff.max(),causal_diff.max())*1.1
    plt.xlim([min_val,max_val])
    plt.ylim([min_val,max_val])
    plt.plot(np.linspace(min_val,max_val,10),np.linspace(min_val,max_val,10),
           linestyle='--',color='grey')
    plt.xlabel('{}'.format(eval_metric.upper()) + r'$^{rand}-$' + '{}'.format(eval_metric.upper()) + r'$^{baseline}$',fontsize=16)
    plt.ylabel('{}'.format(eval_metric.upper()) + r'$^{causal}-$' + '{}'.format(eval_metric.upper()) + r'$^{baseline}$',fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    sns.despine()

    if save_path is not None:
        plt.savefig(save_path,dpi=500,bbox_inches='tight')

    plt.show()
    
model_color = {'gat': 'green', 'gatv2': 'blue', 'transformer': 'red'}

def plot_attention(all_results,eval_set,eval_metric,save_path=None): #,base_results_df=None):
    
    eval_df = all_results[all_results['set'] == eval_set]
    
    base_metric = eval_df[eval_df['attn'] == 0][eval_metric].mean()
                          
    plt.figure(figsize=(4,5))
    for model,model_df in eval_df.groupby('model'):

        model_df = model_df.sort_values('attn',ascending=True)
        model_df = model_df.groupby(['base','attn']).mean('acc').reset_index()

        plt.plot(model_df[~model_df['base']]['attn'],model_df[~model_df['base']][eval_metric],
               color=model_color[model],label='{} + causal'.format(model),linewidth=5)
        plt.plot(model_df[model_df['base']]['attn'],model_df[model_df['base']][eval_metric],
               color=model_color[model],label=model,linewidth=5,linestyle='dotted')
        
        print('{} AUC:'.format(model),integrate(model_df[~model_df['base']]['attn'].values,
                                                model_df[~model_df['base']][eval_metric].values-base_metric))
        
        print('{} (base) AUC:'.format(model),integrate(model_df[model_df['base']]['attn'].values,
                                                       model_df[model_df['base']][eval_metric].values-base_metric))
        
    # if base_results_df is not None:
    plt.axhline(y=base_metric,color='black',linestyle='--',linewidth=2,label='GCN without rewiring')
        
    plt.legend(fontsize=16,bbox_to_anchor=(1,1),frameon=False)

    plt.title(eval_set.upper(),fontsize=20)

    plt.xlabel('Attention Threshold',fontsize=24)
    plt.ylabel(eval_metric.upper(),fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    plt.locator_params(axis='x',nbins=5)
    plt.locator_params(axis='y',nbins=5)
        
    sns.despine()

    if save_path is not None:
        plt.savefig(save_path,dpi=500,bbox_inches='tight')
    plt.show()
    
    
def integrate(x, y):
    
    sm = 0
    for i in range(1, len(x)):
        h = x[i] - x[i-1]
        sm += h * (y[i-1] + y[i]) / 2
    
    return sm