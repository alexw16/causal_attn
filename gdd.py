import networkx as nx
import numpy as np

from utils import *

from torch_geometric.utils import to_dense_adj
from netrd.distance import GraphDiffusion

def networkx_from_edge_index(edge_index):
  
  A = to_dense_adj(edge_index).data.numpy().squeeze()
  A =+ A.T
  A = A.astype(bool).astype(float)
  G = nx.from_numpy_matrix(A)
  
  return G

def calculate_edge_deletion_gdd(edge_index):
    
  gdd = GraphDiffusion()
  
  G = networkx_from_edge_index(edge_index)
  
  edge_list = []
  gdd_list = []
  inds = list(np.arange(edge_index.shape[1]))
  
  # iterate through edges
  for i in range(edge_index.shape[1]):
    if i == 0:
      inds2keep = inds[1:]
    elif i == edge_index.shape[1]-1:
      inds2keep = inds[0:-1]
    else:
      inds2keep = inds[0:i] + inds[i+1:]
      
    remaining_edge_index = edge_index[:,inds2keep]
    G_pruned = networkx_from_edge_index(remaining_edge_index)
    
    edge_list.append(edge_index[:,i].data.numpy())
    gdd_list.append(gdd.dist(G,G_pruned,resolution=100))
    
    if i % 100 == 0:
      print(i)
    
  return np.array(gdd_list)

def main():
  
  dataset_name = 'Cora'
  train_loader,valid_loader,test_loader = load_dataloader(dataset_name,batch_size=5000)
  edge_index = train_loader.data.edge_index
  
  gdd = calculate_edge_deletion_gdd(edge_index)
  # gdd = np.random.random((edge_index.shape[1]))
  data = np.concatenate([edge_index.data.numpy(),np.expand_dims(gdd,0)])
  
  results_dir = '/home/sandbox/workspace/sequence-graphs/results/graph_properties'
  results_path = os.path.join(results_dir,'{}.gdd.txt'.format(dataset_name))
  np.savetxt(results_path,data,delimiter='\t')
  
if __name__ == "__main__":
    main()
    os._exit(1)