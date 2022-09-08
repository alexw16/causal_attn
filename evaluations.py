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

def evaluate_single_model(model_type,evaluator,model,dataloader,device=0,task='gpp',mask_attr=None):
    
    # _,preds = run_epoch_dataloader(0,model,dataloader,model_type=model_type,
    #                              device=device,verbose=False,train=False,task=task,
    #                              mask_attr=mask_attr)

    if task != 'lpp':
        y_pred_list = []
        y_true_list = []
        for batch in dataloader:
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
        y_pred = y_pred.data.numpy()
        
        y_true = torch.cat(y_true_list)
        y_true = y_true.unsqueeze(-1) if y_true.dim() == 1 else y_true
        y_true = y_true.data.numpy()

        result_dict = evaluator.eval({"y_true": y_true,"y_pred": y_pred})

    elif task == 'lpp':
        y_pred_pos = torch.cat([preds[batch.edge_label.nonzero().squeeze()] for batch in dataloader]).squeeze()
        y_pred_neg = torch.cat([preds[(batch.edge_label == 0).nonzero().squeeze()] for batch in dataloader]).squeeze()

        result_dict = evaluator.eval({"y_pred_pos": y_pred_pos,"y_pred_neg": y_pred_neg})

    torch.cuda.empty_cache() 

    return result_dict

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

def evaluate_models_gcn_base(dataset_name,evaluator,save_dir,params_dict,
                    eval_metric='acc',device=0,suffix='interv',task='npp',batch_size=5000,attn_thresh=0.1,trial_no=0,
                    gcn_dim_hidden=20):
        
    orig_save_dir = os.path.join(save_dir,'models')
    rewire_save_dir = os.path.join(save_dir,'models_rewire')
    
    data_dict = {k: [] for k in ['model','base','n_layers','dim_hidden',
                               'heads','lc','ni','set',eval_metric]}
    
    for model_base in params_dict['model']:
        model_type = '{}.{}'.format(model_base,suffix)
    
        for dim_hidden in params_dict['hd']:
            for n_layers in params_dict['nl']:
                  
                # reload dataloader
                train_loader,valid_loader,test_loader = load_dataloader(dataset_name,batch_size=batch_size,shuffle_train=False)
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
                    gcn_dim_hidden=20):
    
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
                    train_loader,valid_loader,test_loader = load_dataloader(dataset_name,batch_size=batch_size,shuffle_train=False)
                    dim_in,dim_out,edge_dim,_ = get_dataset_params(dataset_name,valid_loader,dim_hidden)
                    n_embeddings = valid_loader.data.num_nodes if dataset_name == 'ogbl-ddi' else None

                    # graph attention network
                    model = load_model(dataset_name,model_type,heads,dim_in,dim_hidden,dim_out,
                                     n_layers,edge_dim,orig_save_dir,base=True,lc=None,ni=None,n_embeddings=n_embeddings)
                    rewired_train_loader = generate_rewired_dataloader(model,train_loader,attn_thresh=attn_thresh,
                                                                     batch_size=batch_size,shuffle=False,verbose=False)
                    rewired_valid_loader = generate_rewired_dataloader(model,valid_loader,attn_thresh=attn_thresh,
                                                                     batch_size=batch_size,shuffle=False,verbose=False)
                    rewired_test_loader = generate_rewired_dataloader(model,test_loader,attn_thresh=attn_thresh,
                                                                    batch_size=batch_size,shuffle=False,verbose=False)

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
                            train_loader,valid_loader,test_loader = load_dataloader(dataset_name,batch_size=batch_size,shuffle_train=False)

                            # graph attention network (causal)
                            model = load_model(dataset_name,model_type,heads,dim_in,dim_hidden,dim_out,
                                             n_layers,edge_dim,orig_save_dir,base=False,lc=lc,ni=ni,n_embeddings=n_embeddings)
                            rewired_train_loader = generate_rewired_dataloader(model,train_loader,attn_thresh=attn_thresh,
                                                                             batch_size=batch_size,shuffle=False,verbose=False)
                            rewired_valid_loader = generate_rewired_dataloader(model,valid_loader,attn_thresh=attn_thresh,
                                                                             batch_size=batch_size,shuffle=False,verbose=False)
                            rewired_test_loader = generate_rewired_dataloader(model,test_loader,attn_thresh=attn_thresh,
                                                                            batch_size=batch_size,shuffle=False,verbose=False)

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

def plot_comparisons(results_df,eval_set,eval_metric='acc',alpha_feat='lc',save_path=None,plot=True):
    
    plot_dict = {k: [] for k in ['x','y','model','params']}
    results_df = results_df[results_df['set'] == eval_set]
    for model_base,group_df in results_df.groupby('model'):
        for params,param_df in group_df.groupby(['n_layers','dim_hidden','heads']):
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
        for model,model_df in plot_df.groupby('model'):

            alpha_scale = np.log(1+model_df[alpha_feat]/model_df[alpha_feat].min())
            plt.scatter(model_df['x'],model_df['y'], 
                      c=model_color[model],label=model,edgecolors='black',
                      s=150*(1+model_df[alpha_feat])/(1+model_df[alpha_feat]).max())
                      #alpha=(1+model_df[alpha_feat])/(1+model_df[alpha_feat]).max(),

        min_value = plot_df[['x','y']].values.min()*0.95
        max_value = plot_df[['x','y']].values.max()*1.05

        plt.xlim([min_value,max_value])
        plt.ylim([min_value,max_value])

        plt.plot([min_value,max_value],[min_value,max_value],linestyle='--',linewidth=1,color='grey')
        leg = plt.legend(fontsize=16,bbox_to_anchor=(1,1),frameon=False)

        for lh in leg.legendHandles: 
            lh.set_alpha(1)
            lh._sizes = [100] 

        ticks = (np.linspace(min_value,max_value,20)*20).astype(int)/20
        ticks = sorted(list(set(ticks)))
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel('Baseline\n{}'.format(eval_metric.upper()),fontsize=18)
        plt.ylabel('Causal Attention\n{}'.format(eval_metric.upper()),fontsize=18)

        plt.title(eval_set.upper(),fontsize=18)

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

def plot_attention(all_results,eval_set,eval_metric,save_path=None,base_results_df=None):
    
    eval_df = all_results[all_results['set'] == eval_set]

    plt.figure(figsize=(4,5))
    for model,model_df in eval_df.groupby('model'):

        model_df = model_df.sort_values('attn',ascending=True)
        model_df = model_df.groupby(['base','attn']).mean('acc').reset_index()

        plt.plot(model_df[~model_df['base']]['attn'],model_df[~model_df['base']][eval_metric],
               color=model_color[model],label='{} + causal'.format(model),linewidth=5)
        plt.plot(model_df[model_df['base']]['attn'],model_df[model_df['base']][eval_metric],
               color=model_color[model],label=model,linewidth=5,linestyle='dotted')

    if base_results_df is not None:
        plt.axhline(y=base_results_df.loc[eval_set][eval_metric],
                    color='black',linestyle='--',linewidth=2,label='GCN without rewiring')
        
    plt.legend(fontsize=16,bbox_to_anchor=(1,1),frameon=False)

    plt.title(eval_set.upper(),fontsize=20)

    plt.xlabel('Attention Threshold',fontsize=18)
    plt.ylabel(eval_metric.upper(),fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    sns.despine()

    if save_path is not None:
        plt.savefig(save_path,dpi=500,bbox_inches='tight')
    plt.show()