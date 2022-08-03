import torch
import torch.nn as nn
from torch_geometric.nn.conv import GATConv,GATv2Conv,TransformerConv
from torch_geometric.utils import remove_self_loops

class CausalGAT(nn.Module):
    def __init__(self,model_type,dim_in,dim_hidden,dim_out,heads=3,n_layers=1,edge_dim=None):
        super(CausalGAT, self).__init__()

        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.heads = heads
        self.n_layers = n_layers

        self.linearSeq = nn.Sequential(
                          nn.Linear(dim_in,dim_hidden,bias=True),
                          nn.LeakyReLU()) #,
                          # nn.Linear(dim_hidden,dim_hidden,bias=True),
                          # nn.LeakyReLU())
        
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
            
            setattr(self,'gat_{}'.format(n+1),gat)

        self.linear_final = nn.Linear(dim_hidden*2,dim_out,bias=True)

        self.decay_rate = nn.Parameter(torch.ones(1),requires_grad=True)

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