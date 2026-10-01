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
    def __init__(self, command_dims, proprio_dims, action_dims, token_dim=128, hidden=(512, 512, 256), variant='shared'):
        super().__init__()
        if variant not in ('shared', 'independent', 'no_align'):
            raise ValueError(variant)
        self.variant = variant
        self.robots = tuple(action_dims)
        self.constructor = dict(command_dims=command_dims, proprio_dims=proprio_dims, action_dims=action_dims,
                                token_dim=token_dim, hidden=list(hidden), variant=variant)
        groups = self.robots if variant == 'independent' else ('shared',)
        self.encoders = nn.ModuleDict({g: nn.ModuleDict({s: mlp(command_dims[s], [256, token_dim]) for s in SOURCES}) for g in groups})
        self.backbones = nn.ModuleDict({g: mlp(token_dim*2, hidden) for g in groups})
        self.proprio = nn.ModuleDict({r: mlp(proprio_dims[r], [256, token_dim]) for r in self.robots})
        self.heads = nn.ModuleDict({r: nn.Linear(hidden[-1], action_dims[r]) for r in self.robots})
        self.critics = nn.ModuleDict({r: nn.Sequential(mlp(proprio_dims[r]+2*action_dims[r]+9, [512, 256]), nn.Linear(256, 1)) for r in self.robots})
        self.log_std = nn.ParameterDict({r: nn.Parameter(torch.zeros(action_dims[r])) for r in self.robots})
        for head in self.heads.values():
            nn.init.orthogonal_(head.weight, gain=.01)
            nn.init.zeros_(head.bias)

    def tokens(self, robot, commands):
        group = robot if self.variant == 'independent' else 'shared'
        return torch.stack([self.encoders[group][s](commands[s]) for s in SOURCES], 1)

    def actor(self, robot, proprio, source, commands):
        tokens = self.tokens(robot, commands)
        chosen = tokens[torch.arange(len(source), device=source.device), source]
        group = robot if self.variant == 'independent' else 'shared'
        hidden = self.backbones[group](torch.cat((chosen, self.proprio[robot](proprio)), -1))
        mean = self.heads[robot](hidden)
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
