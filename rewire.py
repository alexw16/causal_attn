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
from run import instantiate_model

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
    parser.add_argument('-at', dest='attn_thresh',type=float,default=0.1)
    parser.add_argument('-nt', dest='n_trials',type=int,default=5)
    parser.add_argument('-ghd', dest='gcn_dim_hidden',type=int,default=200)

    args = parser.parse_args()
    
    model_dir = os.path.join(args.save_dir,'models')

    rewire_model_dir = os.path.join(args.save_dir,'models_rewire')
    if not os.path.exists(rewire_model_dir):
        os.mkdir(rewire_model_dir)
        
    print('Loading data...')
    
    if 'ogbg-mol' in args.dataset:
        batch_size=512
    elif 'ogbn' in args.dataset or args.dataset in ['Cora','CiteSeer','PubMed']:
        batch_size=5000
    elif 'ogbl' in args.dataset:
        batch_size=10000
        
    train_loader,valid_loader,_ = load_dataloader(args.dataset,batch_size=batch_size,shuffle_train=False)
    dim_in,dim_out,edge_dim,pred_criterion = get_dataset_params(args.dataset,train_loader,args.dim_hidden)
    
    print('Initializing Models...')
    np.random.seed(1)
    torch.manual_seed(1)

    if 'ogbn' in args.dataset or args.dataset in ['Cora','CiteSeer','PubMed']:
        task = 'npp'
    elif 'ogbg' in args.dataset:
        task = 'gpp'
    elif 'ogbl' in args.dataset:
        task = 'lpp'
        
    initial_learning_rate=0.001
    beta_1=0.9
    beta_2=0.999
    
    start = time.time()
    
    if args.device != 'cpu':
        device = int(args.device)
        torch.cuda.set_device(device)
    else:
        device = 'cpu'
        
    print('Device: {}'.format(device))
    
    
    for trial_no in range(args.n_trials):
        
        model_file_name = '{}.trial{}.{}hd.nl{}.gcn.base.pt'.format(args.model_type,
                                                                    trial_no,args.dim_hidden,
                                                                    args.n_layers)
        if not os.path.exists(os.path.join(rewire_model_dir,model_file_name)):
            
            print('Training baseline GCN model...')
            train_loader,valid_loader,_ = load_dataloader(args.dataset,batch_size=batch_size,shuffle_train=False)
            n_embeddings = train_loader.data.num_nodes if args.dataset == 'ogbl-ddi' else None
            model_gcn = instantiate_model(args.dataset,'gcnconv',dim_in,args.gcn_dim_hidden,dim_out,
                                          heads=args.K,n_layers=args.n_layers,edge_dim=edge_dim,
                                          n_embeddings=n_embeddings,seed=trial_no)

            optimizer = torch.optim.Adam(params=model_gcn.parameters(), 
                            lr=initial_learning_rate, betas=(beta_1, beta_2))        
            train_model_dataloader(model_gcn,train_loader,args.model_type,optimizer,device,
                                   num_epochs=args.num_epochs_tuning,pred_criterion=pred_criterion,
                                   early_stop=args.early_stop,tol=args.tol,verbose=True,
                                   intervention_loss=False,task=task,valid_dataloader=valid_loader)

            torch.save(model_gcn.state_dict(),os.path.join(rewire_model_dir,model_file_name))

        # baseline model
        model_file_name = '{}.base.{}heads.{}hd.nl{}.pt'.format(args.model_type,args.K,
                                                           args.dim_hidden,args.n_layers)
        rewire_model_file_name = '{}.trial{}.base.{}heads.{}hd.nl{}.gcn.thresh{}.pt'.format(args.model_type,trial_no,args.K,
                                                           args.dim_hidden,args.n_layers,args.attn_thresh)

        if not os.path.exists(os.path.join(rewire_model_dir,rewire_model_file_name)):

            # rewiring graph
            train_loader,_,_ = load_dataloader(args.dataset,batch_size=batch_size,shuffle_train=False)
            n_embeddings = train_loader.data.num_nodes if args.dataset == 'ogbl-ddi' else None
            model = instantiate_model(args.dataset,args.model_type,dim_in,args.dim_hidden,dim_out,
                                          heads=args.K,n_layers=args.n_layers,edge_dim=edge_dim,
                                          n_embeddings=n_embeddings,seed=trial_no)
            model.load_state_dict(torch.load(os.path.join(model_dir,model_file_name)))

            rewired_train_loader = generate_rewired_dataloader(model,train_loader,args.attn_thresh,
                                                               batch_size=batch_size,
                                                               shuffle=True,verbose=True)
            rewired_valid_loader = generate_rewired_dataloader(model,valid_loader,args.attn_thresh,
                                                               batch_size=batch_size,
                                                               shuffle=True,verbose=True)

            print('Training GCN model: rewired graph (baseline attention)...')
            n_embeddings = train_loader.data.num_nodes if args.dataset == 'ogbl-ddi' else None
            model_gcn = instantiate_model(args.dataset,'gcnconv',dim_in,args.gcn_dim_hidden,dim_out,
                                          heads=args.K,n_layers=args.n_layers,edge_dim=edge_dim,
                                          n_embeddings=n_embeddings,seed=trial_no)
            optimizer = torch.optim.Adam(params=model_gcn.parameters(), 
                            lr=initial_learning_rate, betas=(beta_1, beta_2))

            train_start = time.time()
            train_model_dataloader(model_gcn,rewired_train_loader,args.model_type,optimizer,device,
                                   num_epochs=args.num_epochs_tuning,pred_criterion=pred_criterion,
                                   early_stop=args.early_stop,tol=args.tol,verbose=True,
                                   intervention_loss=False,task=task,valid_dataloader=rewired_valid_loader)
            # save model
            torch.save(model_gcn.state_dict(),os.path.join(rewire_model_dir,rewire_model_file_name))
            np.savetxt(os.path.join(rewire_model_dir,'time.' + rewire_model_file_name),
                       np.array([time.time()-train_start]))

        model_file_name = '{}.{}heads.{}hd.nl{}.lc{}.ni{}.pt'.format(args.model_type,args.K,
                                                                    args.dim_hidden,args.n_layers,
                                                                    args.lam_causal,
                                                                    args.n_interventions)
        rewire_model_file_name = '{}.trial{}.{}heads.{}hd.nl{}.lc{}.ni{}.gcn.thresh{}.pt'.format(args.model_type,trial_no,args.K,
                                                                    args.dim_hidden,args.n_layers,
                                                                    args.lam_causal,
                                                                    args.n_interventions,args.attn_thresh)

        if not os.path.exists(os.path.join(rewire_model_dir,rewire_model_file_name)):

            # rewiring graph
            train_loader,valid_loader,_ = load_dataloader(args.dataset,batch_size=batch_size,shuffle_train=False)
            n_embeddings = train_loader.data.num_nodes if args.dataset == 'ogbl-ddi' else None
            model = instantiate_model(args.dataset,args.model_type,dim_in,args.dim_hidden,dim_out,
                                          heads=args.K,n_layers=args.n_layers,edge_dim=edge_dim,
                                          n_embeddings=n_embeddings,seed=trial_no)
            model.load_state_dict(torch.load(os.path.join(model_dir,model_file_name)))

            rewired_train_loader = generate_rewired_dataloader(model,train_loader,args.attn_thresh,
                                                               batch_size=batch_size,
                                                               shuffle=True,verbose=True)
            rewired_valid_loader = generate_rewired_dataloader(model,valid_loader,args.attn_thresh,
                                                               batch_size=batch_size,
                                                               shuffle=True,verbose=True)
            
            print('Training GCN model: rewired graph (causal attention)...')
            n_embeddings = train_loader.data.num_nodes if args.dataset == 'ogbl-ddi' else None
            model_gcn = instantiate_model(args.dataset,'gcnconv',dim_in,args.gcn_dim_hidden,dim_out,
                                          heads=args.K,n_layers=args.n_layers,edge_dim=edge_dim,
                                          n_embeddings=n_embeddings,seed=trial_no)
            optimizer = torch.optim.Adam(params=model_gcn.parameters(), 
                            lr=initial_learning_rate, betas=(beta_1, beta_2))

            train_start = time.time()
            train_model_dataloader(model_gcn,rewired_train_loader,args.model_type,optimizer,device,
                                   num_epochs=args.num_epochs_tuning,pred_criterion=pred_criterion,
                                   early_stop=args.early_stop,tol=args.tol,verbose=True,
                                   intervention_loss=False,task=task,valid_dataloader=rewired_valid_loader)

            # save model
            torch.save(model_gcn.state_dict(),os.path.join(rewire_model_dir,rewire_model_file_name))
            np.savetxt(os.path.join(rewire_model_dir,'time.' + rewire_model_file_name),
                       np.array([time.time()-train_start]))

    print('Total Time: {} seconds'.format(time.time()-start))
    
if __name__ == "__main__":
    main()
    os._exit(1)
  