"""Small modality-fusion Transformer; consumes exactly the existing flattened observations."""
import torch
from torch import nn
from torch.nn import functional as F


class AttentionBlock(nn.Module):
    def __init__(self,width,heads,ff_dim):
        super().__init__();self.heads=heads;self.width=width
        self.norm1=nn.LayerNorm(width);self.qkv=nn.Linear(width,3*width);self.proj=nn.Linear(width,width)
        self.norm2=nn.LayerNorm(width);self.ff=nn.Sequential(nn.Linear(width,ff_dim),nn.GELU(),nn.Linear(ff_dim,width))
    def forward(self,x):
        b,n,d=x.shape
        q,k,v=self.qkv(self.norm1(x)).reshape(b,n,3,self.heads,d//self.heads).permute(2,0,3,1,4).unbind(0)
        attended=F.scaled_dot_product_attention(q,k,v,dropout_p=0.).transpose(1,2).reshape(b,n,d)
        x=x+self.proj(attended)
        return x+self.ff(self.norm2(x))


class FusionTransformer(nn.Module):
    def __init__(self,input_dims,output_dim,d_model=256,nhead=4,layers=2,ff_dim=512):
        super().__init__()
        if d_model%nhead or min(*input_dims,output_dim,d_model,nhead,layers,ff_dim)<=0:raise ValueError('Invalid Transformer dimensions')
        self.input_dims=list(input_dims)
        self.projections=nn.ModuleList([nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,d_model)) for dim in input_dims])
        self.cls=nn.Parameter(torch.zeros(1,1,d_model));self.types=nn.Parameter(torch.empty(1,len(input_dims)+1,d_model))
        nn.init.normal_(self.types,std=.02);nn.init.normal_(self.cls,std=.02)
        self.blocks=nn.Sequential(*[AttentionBlock(d_model,nhead,ff_dim) for _ in range(layers)])
        self.norm=nn.LayerNorm(d_model);self.output=nn.Linear(d_model,output_dim) if output_dim!=d_model else nn.Identity()
    def forward(self,flat):
        parts=flat.split(self.input_dims,dim=-1)
        tokens=torch.stack([p(x) for p,x in zip(self.projections,parts)],dim=1)
        tokens=torch.cat([self.cls.expand(flat.shape[0],-1,-1),tokens],dim=1)+self.types
        return self.output(self.norm(self.blocks(tokens)[:,0]))
