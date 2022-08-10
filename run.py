import pandas as pd
import numpy as np
import argparse
import time
from scipy.io import mmread
import os

import torch.nn as nn

from train import *
from utils import *
from models import *
from causal import *
        
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-ds', dest='dataset',type=str)
    parser.add_argument('-sd', dest='save_dir',type=str)
    parser.add_argument('-mt', dest='model_type',type=str)
    parser.add_argument('-hd', dest='dim_hidden',type=int)
    parser.add_argument('-K', dest='K',type=int)
    parser.add_argument('-nl', dest='n_layers',type=int,default=1)
    parser.add_argument('-ne', dest='num_epochs',type=int)
    parser.add_argument('-net', dest='num_epochs_tuning',type=int)
    parser.add_argument('-d', dest='device',default='cpu')
    parser.add_argument('-lc', dest='lam_causal',type=float,default=1)
    parser.add_argument('-tol', dest='tol',type=float,default=1e-5)
    parser.add_argument('-es', dest='early_stop',type=int,default=1)
    parser.add_argument('-ni', dest='n_interventions',type=int,default=10)

    args = parser.parse_args()
    
    model_dir = os.path.join(args.save_dir,'models')
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
        os.mkdir(model_dir)
        
    print('Loading data...')
    
    if 'ogbg-mol' in args.dataset:
        train_loader,_,_ = load_dataloader(args.dataset,batch_size=512)
        dim_in = args.dim_hidden #train_loader.dataset.data.x.shape[1]
        dim_out = train_loader.dataset.data.y.shape[1]
        edge_dim = args.dim_hidden #train_loader.dataset.data.edge_attr.shape[1]
        
        if any(x in args.dataset for x in ['molesol','molfreesolv','mollipo']):
            pred_criterion = nn.MSELoss(reduction='none')
        else:
            pred_criterion = nn.BCELoss(reduction='none')
    
    elif 'ogbn' in args.dataset or args.dataset in ['Cora','CiteSeer','PubMed']:
        train_loader,_,_ = load_dataloader(args.dataset,batch_size=512)
        dim_in = train_loader.data.x.shape[1]
        dim_out = train_loader.data.y.long().data.numpy().max()+1
        edge_dim = train_loader.data.edge_attr.shape[1] if train_loader.data.edge_attr is not None else None
        pred_criterion = nn.CrossEntropyLoss(reduction='none')
        
    else:
        train_idx,valid_idx,test_idx = load_ogb_splits(args.dataset)
        X,edge_indices,Y,edge_attr,addl_params = load_ogb_data(args.dataset)

        edge_dim = addl_params['edge_dim']
        dim_in = addl_params['dim_in']
        dim_out = addl_params['dim_out']
        pred_criterion = addl_params['pred_criterion']
    
    print('Initializing Models...')
    np.random.seed(1)
    torch.manual_seed(1)
    
    if 'ogbn' in args.dataset or args.dataset in ['Cora','CiteSeer','PubMed']:
        model = GATNode(args.model_type,dim_in,args.dim_hidden,dim_out,
                          heads=args.K,n_layers=args.n_layers,edge_dim=edge_dim)
        model_causal = GATNode(args.model_type,dim_in,args.dim_hidden,dim_out,
                                 heads=args.K,n_layers=args.n_layers,edge_dim=edge_dim)
        task = 'npp'
        
    elif 'ogbg-mol' in args.dataset:
        model = GATMolecule(args.model_type,dim_in,args.dim_hidden,dim_out,
                          heads=args.K,n_layers=args.n_layers,edge_dim=edge_dim)
        model_causal = GATMolecule(args.model_type,dim_in,args.dim_hidden,dim_out,
                                 heads=args.K,n_layers=args.n_layers,edge_dim=edge_dim)
        task = 'gpp'
        
    elif 'ogbg' in args.dataset:
        model = GATGraph(args.model_type,dim_in,args.dim_hidden,dim_out,
                          heads=args.K,n_layers=args.n_layers,edge_dim=edge_dim)
        model_causal = GATGraph(args.model_type,dim_in,args.dim_hidden,dim_out,
                                 heads=args.K,n_layers=args.n_layers,edge_dim=edge_dim)
        task = 'gpp'
        
    initial_learning_rate=0.01
    beta_1=0.9
    beta_2=0.999
    optimizer = torch.optim.Adam(params=model.parameters(), 
                        lr=initial_learning_rate, betas=(beta_1, beta_2))
    
    start = time.time()
    
    if args.device != 'cpu':
        device = int(args.device)
        torch.cuda.set_device(device)
    else:
        device = 'cpu'
        
    print('Device: {}'.format(device))
    
    print('Training...')
    model_file_name = '{}.init.{}heads.{}hd.nl{}.pt'.format(args.model_type,args.K,
                                                           args.dim_hidden,args.n_layers)
    optimizer_file_name = '{}.init.{}heads.{}hd.nl{}.optimizer.pt'.format(args.model_type,args.K,
                                                           args.dim_hidden,args.n_layers)
    if os.path.exists(os.path.join(model_dir,model_file_name)) \
        and os.path.exists(os.path.join(model_dir,optimizer_file_name)):
        print('Loading initial model + optimizer...')
        model.load_state_dict(torch.load(os.path.join(model_dir,model_file_name)))
        optimizer.load_state_dict(torch.load(os.path.join(model_dir,optimizer_file_name)))
        
    else:
        if 1: #'ogb' in args.dataset:
            train_model_dataloader(model,train_loader,args.model_type,optimizer,device,
                           num_epochs=args.num_epochs,pred_criterion=pred_criterion,
                           intervention_loss=False,early_stop=args.early_stop,
                           tol=args.tol,verbose=True,task=task)

        else:
            train_model(model,X,edge_indices,Y,args.model_type,optimizer,device,
                        node_indices=train_idx,
                        edge_attr=edge_attr,num_epochs=args.num_epochs,
                        intervention_loss=False,
                        early_stop=args.early_stop,tol=args.tol,verbose=True,
                        pred_criterion=pred_criterion)

        # save model + optimizer
        torch.save(model.state_dict(),os.path.join(model_dir,model_file_name))
        torch.save(optimizer.state_dict(),os.path.join(model_dir,optimizer_file_name))
    
    # instantiate new causal model
    model_causal.load_state_dict(model.state_dict().copy())
    model_causal.to(device) # to ensure optimizer on 'device'
    optimizer_causal = torch.optim.Adam(params=model_causal.parameters(), 
                    lr=initial_learning_rate, betas=(beta_1, beta_2))
    optimizer_causal.load_state_dict(optimizer.state_dict())
    
    model_file_name = '{}.base.{}heads.{}hd.nl{}.pt'.format(args.model_type,args.K,
                                                       args.dim_hidden,args.n_layers)
    if not os.path.exists(os.path.join(model_dir,model_file_name)):
        print('Continue training (baseline)...')
        if 1: #'ogb' in args.dataset:
            train_model_dataloader(model,train_loader,args.model_type,optimizer,device,
                           num_epochs=args.num_epochs_tuning,pred_criterion=pred_criterion,
                           intervention_loss=False,early_stop=args.early_stop,
                           tol=args.tol,verbose=True,task=task)
        else:
            train_model(model,X,edge_indices,Y,
                        args.model_type,optimizer,device,
                        node_indices=train_idx,
                        edge_attr=edge_attr,num_epochs=args.num_epochs_tuning,
                        intervention_loss=False,
                        early_stop=args.early_stop,tol=args.tol,verbose=True,
                        pred_criterion=pred_criterion)
            
        # save model
        torch.save(model.state_dict(),os.path.join(model_dir,model_file_name))
    
    print('Continue training (causal)...')
    
    if 1: #'ogb' in args.dataset:
        train_model_dataloader(model_causal,train_loader,args.model_type,optimizer_causal,device,
                               num_epochs=args.num_epochs_tuning,pred_criterion=pred_criterion,
                               early_stop=args.early_stop,tol=args.tol,verbose=True,
                               intervention_loss=True,lam_causal=args.lam_causal,
                               n_interventions_per_node=args.n_interventions,task=task)
    else:
        train_model(model_causal,X,edge_indices,Y,
                    args.model_type,optimizer_causal,device,
                    node_indices=train_idx,
                    edge_attr=edge_attr,num_epochs=args.num_epochs_tuning,
                    intervention_loss=True,
                    lam_causal=args.lam_causal,
                    early_stop=args.early_stop,tol=args.tol,verbose=True,
                    pred_criterion=pred_criterion,
                    n_interventions_per_node=args.n_interventions)

    # save model
    model_file_name = '{}.{}heads.{}hd.nl{}.lc{}.ni{}.pt'.format(args.model_type,args.K,
                                                            args.dim_hidden,args.n_layers,
                                                            args.lam_causal,
                                                            args.n_interventions)
    torch.save(model_causal.state_dict(),os.path.join(model_dir,model_file_name))

    print('Total Time: {} seconds'.format(time.time()-start))
    
if __name__ == "__main__":
    main()
    os._exit(1)
  