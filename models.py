import torch
import torch.nn as nn
from torch_geometric.nn.conv import GCNConv,GATConv,GATv2Conv,TransformerConv
from torch_geometric.utils import remove_self_loops
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from utils import aggregate_using_ptr
      
class GCNConvBase(nn.Module):
    def __init__(self,dim_in,dim_out):
        super(GCNConvBase, self).__init__()

        self.gcn = GCNConv(dim_in,dim_out,add_self_loops=False)
    
    def forward(self,x,edge_index,edge_attr,return_attention_weights):
        return self.gcn(x,edge_index),(None,None)
        
class GATBase(nn.Module):
    def __init__(self,model_type,dim_in,dim_hidden,dim_out,
                 heads=3,n_layers=1,edge_dim=None):
        super(GATBase, self).__init__()

        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.heads = heads
        self.n_layers = n_layers

        self.linearSeq = nn.Sequential(
                          nn.Linear(dim_in,dim_hidden,bias=True),
                          nn.LeakyReLU())
        
        for n in range(self.n_layers):
            if 'gatconv' in model_type:
                gat = GATConv((dim_hidden,dim_hidden),dim_hidden,
                                         heads=heads,add_self_loops=False,
                                         edge_dim=edge_dim,concat=False)
            elif 'gatv2conv' in model_type:
                gat = GATv2Conv((dim_hidden,dim_hidden),dim_hidden,
                                         heads=heads,add_self_loops=False,
                                         edge_dim=edge_dim,concat=False)
            elif 'transformerconv' in model_type:
                gat = TransformerConv(dim_hidden,dim_hidden,
                                         heads=heads,
                                         edge_dim=edge_dim,concat=False)
            elif 'gcnconv' in model_type:
                gat = GCNConvBase(dim_hidden,dim_hidden)
                
            setattr(self,'gat_{}'.format(n+1),gat)
        
    def forward(self):
        pass


class GATNode(GATBase):
    def __init__(self,model_type,dim_in,dim_hidden,dim_out,
                 heads=3,n_layers=1,edge_dim=None):
        super(GATNode, self).__init__(model_type,dim_in,dim_hidden,dim_out,
                 heads,n_layers,edge_dim)
        self.linear_final = nn.Linear(dim_hidden*2,dim_out,bias=True)
        self.leakyrelu = nn.LeakyReLU()
        
    def forward(self,X,edge_indices,edge_attr=None):
        
        h = self.linearSeq(X)
        
        attn_weights_list = []
        for n in range(self.n_layers):
            inp = h if n == 0 else out
            gat = getattr(self,'gat_{}'.format(n+1))
            out,(_,attn_weights) = gat(inp,edge_indices,edge_attr=edge_attr,
                                            return_attention_weights=True)
            attn_weights_list.append(attn_weights)
        out = torch.cat([h,out],1)
        
        out = self.leakyrelu(out)
        out = self.linear_final(out)

        return out,attn_weights_list
    
class GATLink(GATBase):
    def __init__(self,model_type,dim_in,dim_hidden,dim_out,
                 heads=3,n_layers=1,edge_dim=None):
        super(GATLink, self).__init__(model_type,dim_in,dim_hidden,dim_out,
                 heads,n_layers,edge_dim)
        self.gatnode = GATNode(model_type,dim_in,dim_hidden,dim_out,
                 heads,n_layers,edge_dim)
        self.trans_emb = nn.Parameter(torch.ones(dim_hidden),
            requires_grad=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,X,edge_indices,edge_indices_pred,edge_attr=None):
        
        node_emb,attn_weights_list = self.gatnode(X,edge_indices,edge_attr)
        i,j = edge_indices_pred[0],edge_indices_pred[1]
        out = ((node_emb[i] + self.trans_emb)*node_emb[j]).sum(1)
        out = self.sigmoid(out)
        
        return out,attn_weights_list
    
class GATLinkEmbed(GATBase):
    def __init__(self,model_type,dim_in,dim_hidden,dim_out,
                 heads=3,n_layers=1,edge_dim=None,n_embeddings=None):
        super(GATLinkEmbed, self).__init__(model_type,dim_in,dim_hidden,dim_out,
                 heads,n_layers,edge_dim)
        self.embed = nn.Embedding(n_embeddings,dim_hidden)
        self.gatlink = GATLink(model_type,dim_in,dim_hidden,dim_out,
                 heads,n_layers,edge_dim)
        
    def forward(self,X,edge_indices,edge_indices_pred,edge_attr=None):
        
        embed = self.embed(X).squeeze()

        return self.gatlink(embed,edge_indices,edge_indices_pred,edge_attr)
        
class GATGraph(GATBase):
    def __init__(self,model_type,dim_in,dim_hidden,dim_out,
                 heads=3,n_layers=1,edge_dim=None):
        super(GATGraph, self).__init__(model_type,dim_in,dim_hidden,dim_out,
                 heads,n_layers,edge_dim)
        self.linear_final = nn.Linear(dim_hidden*2,dim_out,bias=True)
        self.leakyrelu = nn.LeakyReLU()
        
    def forward(self,X,edge_indices,ptr,edge_attr=None):
        
        h = self.linearSeq(X)
        
        attn_weights_list = []
        for n in range(self.n_layers):
            inp = h if n == 0 else out
            gat = getattr(self,'gat_{}'.format(n+1))
            out,(_,attn_weights) = gat(inp,edge_indices,edge_attr=edge_attr,
                                            return_attention_weights=True)
            attn_weights_list.append(attn_weights)
        out = torch.cat([h,out],1)
        
        out = self.leakyrelu(out)
        out = self.linear_final(out)
        
        out = aggregate_using_ptr(out,ptr)
        
        return out,attn_weights_list
    
class GATMolecule(nn.Module):
    def __init__(self,model_type,dim_in,dim_hidden,dim_out,
                 heads=3,n_layers=1,edge_dim=None):
        super(GATMolecule, self).__init__()
        
        self.atom_encoder = AtomEncoder(emb_dim = dim_hidden)
        self.bond_encoder = BondEncoder(emb_dim = dim_hidden)
        
        self.gatgraph = GATGraph(model_type,dim_hidden,dim_hidden,dim_out,
                                 heads,n_layers,edge_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,X,edge_indices,ptr,edge_attr):
        
        atom_emb = self.atom_encoder(X)
        edge_emb = self.bond_encoder(edge_attr)
                
        out,attn_weights_list = self.gatgraph(atom_emb,edge_indices,ptr,edge_emb)

        return self.sigmoid(out),attn_weights_list
