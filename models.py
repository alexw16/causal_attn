import torch
import torch.nn as nn
from torch_geometric.nn.conv import GCNConv,GATConv,GATv2Conv,TransformerConv
from torch_geometric.utils import remove_self_loops
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.nn import SumAggregation

from utils import aggregate_using_ptr
      
class GCNConvBase(nn.Module):
    def __init__(self,dim_in,dim_out):
        super(GCNConvBase, self).__init__()

        self.gcn = GCNConv(dim_in,dim_out,add_self_loops=False,normalize=False,bias=True)
    
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
        self.add_self_loops = 'self_loop' in model_type

        self.linearSeq = nn.Sequential(
                          nn.Linear(dim_in,dim_hidden,bias=True),
                          nn.LeakyReLU())
        
        for n in range(self.n_layers):
            if 'gatconv' in model_type:
                gat = GATConv((dim_hidden,dim_hidden),dim_hidden,
                              heads=heads,add_self_loops=self.add_self_loops,
                              edge_dim=edge_dim,concat=False)
            elif 'gatv2conv' in model_type:
                gat = GATv2Conv((dim_hidden,dim_hidden),dim_hidden,
                                heads=heads,add_self_loops=self.add_self_loops,
                                edge_dim=edge_dim,concat=False)
            elif 'transformerconv' in model_type:
                gat = TransformerConv(dim_hidden,dim_hidden,heads=heads,
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
            out,(edge_inds,attn_weights) = gat(inp,edge_indices,edge_attr=edge_attr,
                                            return_attention_weights=True)
            if self.add_self_loops:
                edge_inds,attn_weights = remove_self_loops(edge_inds,attn_weights)
                
            attn_weights_list.append(attn_weights)
        out = torch.cat([h,out],1)
        
        out = self.leakyrelu(out)
        out = self.linear_final(out)

        return out,(edge_inds,attn_weights_list)
    
class GATLink(GATBase):
    def __init__(self,model_type,dim_in,dim_hidden,dim_out,
                 heads=3,n_layers=1,edge_dim=None):
        super(GATLink, self).__init__(model_type,dim_in,dim_hidden,dim_out,
                 heads,n_layers,edge_dim)
        self.gatnode = GATNode(model_type,dim_in,dim_hidden,dim_hidden,
                 heads,n_layers,edge_dim)
        self.linear = nn.Linear(dim_hidden*2,dim_out,bias=True)
        self.leakyrelu = nn.LeakyReLU()
        
    def forward(self,X,edge_indices,edge_indices_pred,edge_attr=None):
        
        node_emb,attn_weights_list = self.gatnode(X,edge_indices,edge_attr)
        i,j = edge_indices_pred[0],edge_indices_pred[1]
        out = torch.cat([node_emb[i],node_emb[j]],dim=1)
        out = self.leakyrelu(out)
        out = self.linear(out).squeeze()
        
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
        self.sum_aggr = SumAggregation()
        
    def forward(self,X,edge_indices,ptr,edge_attr=None):
        
        h = self.linearSeq(X)
        
        attn_weights_list = []
        for n in range(self.n_layers):
            inp = h if n == 0 else out
            gat = getattr(self,'gat_{}'.format(n+1))
            out,(edge_inds,attn_weights) = gat(inp,edge_indices,edge_attr=edge_attr,
                                            return_attention_weights=True)
            if self.add_self_loops:
                edge_inds,attn_weights = remove_self_loops(edge_inds,attn_weights)
                
            attn_weights_list.append(attn_weights)
        out = torch.cat([h,out],1)
        
        out = self.sum_aggr(out,ptr=ptr)
        
        out = self.leakyrelu(out)
        out = self.linear_final(out)
        
        return out,(edge_inds,attn_weights_list)
    
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
                
        return self.gatgraph(atom_emb,edge_indices,ptr,edge_emb)

from torch.nn import Linear, BatchNorm1d, Sequential, ReLU
from torch_geometric.nn import global_mean_pool, global_add_pool, GINConv, GATConv
import torch.nn.functional as F

class GATNet(torch.nn.Module):
    def __init__(self, num_features, 
                       num_classes,
                       hidden,
                       head=4,
                       num_fc_layers=2, 
                       num_conv_layers=3, 
                       dropout=0.2):

        super(GATNet, self).__init__()

        self.global_pool = global_add_pool
        self.dropout = dropout
        hidden_in = num_features
        hidden_out = num_classes
   
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()

        for i in range(num_conv_layers):
            self.bns_conv.append(BatchNorm1d(hidden))
            self.convs.append(GATConv(hidden, int(hidden / head), heads=head, dropout=dropout))
        self.bn_hidden = BatchNorm1d(hidden)
        self.bns_fc = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()

        for i in range(num_fc_layers - 1):
            self.bns_fc.append(BatchNorm1d(hidden))
            self.lins.append(Linear(hidden, hidden))
        self.lin_class = Linear(hidden, hidden_out)

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self,x,edge_index,edge_attr=None):
        
        # x = data.x if data.x is not None else data.feat
        # edge_index, batch = data.edge_index, data.batch
        
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        
        attn_weights_list = []
        for i, conv in enumerate(self.convs):
            x = self.bns_conv[i](x)
            out,(edge_inds,attn_weights) = conv(x, edge_index, return_attention_weights=True)
            x = F.relu(out)
            attn_weights_list.append(attn_weights)

        # x = self.global_pool(x, batch)
        for i, lin in enumerate(self.lins):
            x = self.bns_fc[i](x)
            x = F.relu(lin(x))

        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        return x,(edge_index,attn_weights_list)