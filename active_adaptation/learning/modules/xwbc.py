"""Shared or independent cross-embodiment MLPs and clipped PPO primitives."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

SOURCES = ('human', 'g1', 'roban')


def mlp(dim, widths):
    layers = [nn.LayerNorm(dim)]
    for width in widths:
        layers.extend((nn.Linear(dim, width), nn.ELU()))
        dim = width
    return nn.Sequential(*layers)


class XWBCPolicy(nn.Module):
    def __init__(self, command_dims, proprio_dims, action_dims, token_dim=128, hidden=(512, 512, 256), variant='shared', encoder_hidden=None, transformer=None, student_fsq=None, action_decoder=None):
        super().__init__()
        if variant not in ('shared', 'independent', 'no_align'):
            raise ValueError(variant)
        self.variant = variant
        self.robots = tuple(action_dims)
        encoder_hidden = [256] if encoder_hidden is None else list(encoder_hidden)
        self.constructor = dict(command_dims=command_dims, proprio_dims=proprio_dims, action_dims=action_dims,
                                token_dim=token_dim, hidden=list(hidden), variant=variant, encoder_hidden=encoder_hidden)
        groups = self.robots if variant == 'independent' else ('shared',)
        self.encoders = nn.ModuleDict({g: nn.ModuleDict({s: mlp(command_dims[s], [*encoder_hidden, token_dim]) for s in SOURCES}) for g in groups})
        # Parameter-free tail: every command path (DAgger, actor, evaluation) uses FSQ.
        # Appending after construction preserves baseline RNG draws and parameter keys.
        if student_fsq is not None:
            from .student_latent_quantizer import StudentLatentQuantizer
            self.constructor['student_fsq'] = dict(student_fsq)
            for encoders in self.encoders.values():
                for encoder in encoders.values():
                    encoder.add_module('fsq', StudentLatentQuantizer(**student_fsq))
        self.backbones = nn.ModuleDict({g: mlp(token_dim*2, hidden) for g in groups})
        if transformer is not None:
            from active_adaptation.learning.modules.xwbc_transformer import FusionTransformer
            self.constructor['transformer'] = dict(transformer)
            self.backbones = nn.ModuleDict({g:FusionTransformer([token_dim,token_dim],hidden[-1],**transformer) for g in groups})
        self.proprio = nn.ModuleDict({r: mlp(proprio_dims[r], [*encoder_hidden, token_dim]) for r in self.robots})
        self.heads = nn.ModuleDict({r: nn.Linear(hidden[-1], action_dims[r]) for r in self.robots})
        self.critics = nn.ModuleDict({r: nn.Sequential(mlp(proprio_dims[r]+2*action_dims[r]+9, [512, 256]), nn.Linear(256, 1)) for r in self.robots})
        self.log_std = nn.ParameterDict({r: nn.Parameter(torch.zeros(action_dims[r])) for r in self.robots})
        for head in self.heads.values():
            nn.init.orthogonal_(head.weight, gain=.01)
            nn.init.zeros_(head.bias)

        if action_decoder is not None:
            if variant != 'shared':
                raise ValueError('Experimental action decoders require the shared policy')
            from .xwbc_action_decoder import ActionDecoder
            self.constructor['action_decoder'] = dict(action_decoder)
            self.heads = ActionDecoder(hidden[-1], action_dims, **action_decoder)

    def decode_action(self, robot, hidden):
        if hasattr(self.heads, 'predict'):
            return self.heads.predict(robot, hidden)
        return self.heads[robot](hidden)

    def tokens(self, robot, commands):
        group = robot if self.variant == 'independent' else 'shared'
        return torch.stack([self.encoders[group][s](commands[s]) for s in SOURCES], 1)

    def actor(self, robot, proprio, source, commands):
        if getattr(self.heads, 'kind', None) == 'flow':
            raise RuntimeError('Flow endpoints are not Gaussian means; use the staged deterministic action API')
        tokens = self.tokens(robot, commands)
        chosen = tokens[torch.arange(len(source), device=source.device), source]
        group = robot if self.variant == 'independent' else 'shared'
        hidden = self.backbones[group](torch.cat((chosen, self.proprio[robot](proprio)), -1))
        mean = self.decode_action(robot, hidden)
        return Normal(mean, self.log_std[robot].clamp(-5, 1).exp()), tokens

    def value(self, robot, proprio, privileged):
        return self.critics[robot](torch.cat((proprio, privileged), -1)).squeeze(-1)


def alignment_loss(tokens):
    z = F.normalize(tokens, dim=-1)
    return ((z[:, 0]-z[:, 1]).square().mean() + (z[:, 0]-z[:, 2]).square().mean()
            + (z[:, 1]-z[:, 2]).square().mean()) / 3


def gae(rewards, values, next_values, done, terminal, gamma=.99, lam=.95):
    """Time-major GAE: bootstrap time-limit truncations, never cross a reset."""
    advantage = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * (~terminal[t]).float() * next_values[t] - values[t]
        carry = delta + gamma * lam * (~done[t]).float() * carry
        advantage[t] = carry
    return advantage, advantage + values


def ppo_loss(dist, action, old_logp, advantage, clip):
    logp = dist.log_prob(action).sum(-1)
    ratio = (logp-old_logp).exp()
    policy = -torch.minimum(ratio*advantage, ratio.clamp(1-clip, 1+clip)*advantage).mean()
    entropy = dist.entropy().sum(-1).mean()
    approx_kl = ((ratio-1) - (logp-old_logp)).mean()
    return policy, entropy, approx_kl
