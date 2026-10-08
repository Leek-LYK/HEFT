"""Stage-specific control models and bounded DAgger aggregation for XWBC v0.2."""
from __future__ import annotations
import hashlib
import torch
from torch import nn
from torch.distributions import Normal
from active_adaptation.learning.ppo.common import make_mlp

class NativeTeacher(nn.Module):
    """One native robot, privileged actor; no shared parameters or source alignment."""
    def __init__(self,robot,command_dim,proprio_dim,privileged_dim,action_dim,hidden=(1024,1024,512),transformer=None):
        super().__init__()
        self.robot=robot
        self.constructor=dict(robot=robot,command_dim=command_dim,proprio_dim=proprio_dim,
                              privileged_dim=privileged_dim,action_dim=action_dim,hidden=list(hidden))
        dim=command_dim+proprio_dim+privileged_dim
        self.actor_body=make_mlp(hidden)
        if transformer is not None:
            from active_adaptation.learning.modules.xwbc_transformer import FusionTransformer
            self.constructor['transformer']=dict(transformer)
            self.actor_body=FusionTransformer([proprio_dim,privileged_dim,command_dim],hidden[-1],**transformer)
        self.head=nn.Linear(hidden[-1],action_dim)
        self.critic=nn.Sequential(make_mlp([1024,512,256]),nn.Linear(256,1))
        self.log_std=nn.Parameter(torch.zeros(action_dim))
        with torch.no_grad():
            dummy=torch.zeros(1,dim);self.head(self.actor_body(dummy));self.critic(dummy)
        nn.init.orthogonal_(self.head.weight,gain=.01);nn.init.zeros_(self.head.bias)

    def distribution(self,proprio,privileged,native_command):
        x=torch.cat((proprio,privileged,native_command),-1)
        return Normal(self.head(self.actor_body(x)),self.log_std.clamp(-5,1).exp())

    def value(self,proprio,privileged,native_command):
        return self.critic(torch.cat((proprio,privileged,native_command),-1)).squeeze(-1)


def student_features(model,robot,proprio,command,source):
    token=model.encoders['shared'][source](command)
    return model.backbones['shared'](torch.cat((token,model.proprio[robot](proprio)),-1))


def student_mean(model,robot,proprio,command,source):
    """Deterministic executed action; a flow endpoint is not a distribution mean."""
    hidden=student_features(model,robot,proprio,command,source)
    if hasattr(model.heads,'predict'):
        return model.heads.predict(robot,hidden)
    return model.heads[robot](hidden)


def configure_student(model,stage):
    model.requires_grad_(False);model.eval()
    if stage=='B':
        modules=[model.encoders['shared']['human'],model.backbones['shared'],model.proprio,model.heads]
    elif stage=='C':
        if hasattr(model.heads,'predict'):raise ValueError('Experimental action decoders support B/E only')
        modules=[model.encoders['shared']['g1'],model.encoders['shared']['roban']]
    else:raise ValueError(stage)
    for module in modules:module.requires_grad_(True);module.train()
    return [p for p in model.parameters() if p.requires_grad]


def action_mse(student,teacher):
    # Per sample and per valid action dimension, within one target robot.
    return (student-teacher.detach()).square().mean()


def stage_c_loss(model,robot,proprio,commands,anchor_action,token_weight,action_weight):
    with torch.no_grad():anchor=model.encoders['shared']['human'](commands['human'])
    token_loss=anchor.new_zeros(());act_loss=anchor.new_zeros(())
    for source in ['g1','roban']:
        token=model.encoders['shared'][source](commands[source])
        token_loss=token_loss+(token-anchor).square().mean()  # raw token, including amplitude
        hidden=model.backbones['shared'](torch.cat((token,model.proprio[robot](proprio)),-1))
        act_loss=act_loss+action_mse(model.heads[robot](hidden),anchor_action)
    return token_weight*token_loss+action_weight*act_loss,token_loss,act_loss


def frozen_digest(model):
    h=hashlib.sha256()
    trainable={k for k,p in model.named_parameters() if p.requires_grad}
    for key,tensor in sorted(model.state_dict().items()):
        if key not in trainable:
            h.update(key.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class Reservoir:
    """Uniform bounded reservoir across all visited states, with batch-safe collisions."""
    def __init__(self,capacity,seed=0):
        self.capacity=int(capacity);self.seen=0;self.data={}
        self.generator=torch.Generator(device='cpu').manual_seed(seed)

    @property
    def size(self):return min(self.capacity,self.seen)

    def add(self,rows):
        rows={k:v.detach().cpu() for k,v in rows.items()}
        n=len(next(iter(rows.values())))
        if any(len(v)!=n for v in rows.values()):raise ValueError('Replay fields have different row counts')
        if not self.data:
            self.data={k:torch.empty((self.capacity,*v.shape[1:]),dtype=v.dtype) for k,v in rows.items()}
        indices=torch.arange(self.seen+1,self.seen+n+1)
        destination=(torch.rand(n,generator=self.generator,dtype=torch.float64)*indices).long()
        destination=torch.where(indices<=self.capacity,indices-1,destination)
        valid=destination<self.capacity
        # Repeated reservoir slots use the last arrival, exactly as sequential replacement.
        arrivals=torch.arange(n)[valid]
        last=torch.full((self.capacity,),-1,dtype=torch.long)
        last.scatter_reduce_(0,destination[valid],arrivals,reduce='amax',include_self=True)
        slots=(last>=0).nonzero().flatten()
        for key,value in rows.items():self.data[key][slots]=value[last[slots]]
        self.seen+=n

    def sample(self,size,device):
        if self.size==0:raise ValueError('Empty replay reservoir')
        idx=torch.randint(self.size,(size,),generator=self.generator)
        return {k:v[idx].to(device) for k,v in self.data.items()}

    def state_dict(self):
        return dict(capacity=self.capacity,seen=self.seen,data={k:v[:self.size].clone() for k,v in self.data.items()},rng=self.generator.get_state())

    def load_state_dict(self,state):
        if state['capacity']!=self.capacity:raise ValueError('Replay capacity changed')
        self.seen=state['seen'];self.generator.set_state(state['rng'].cpu())
        self.data={k:torch.empty((self.capacity,*v.shape[1:]),dtype=v.dtype) for k,v in state['data'].items()}
        for k,v in state['data'].items():self.data[k][:len(v)]=v.cpu()
