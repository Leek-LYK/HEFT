"""Robot-conditioned action decoders for controlled MLP versus flow experiments."""
from __future__ import annotations
import math
import torch
from torch import nn


class ActionDecoder(nn.Module):
    """Shared nonlinear decoder with native-dimension robot interfaces.

    Flow predictions are deterministic zero-initialized Euler endpoints, not
    Gaussian means. Matching noise has a private checkpointed RNG per device.
    """
    def __init__(self, condition_dim, action_dims, kind, width=256, steps=4,
                 action_embed=64, time_embed=32, robot_embed=16,
                 init_seed=12345, noise_seed=314159):
        super().__init__()
        if kind not in ('flow', 'mlp'):
            raise ValueError('Unknown action decoder: '+kind)
        if min(width, steps, action_embed, time_embed, robot_embed) <= 0 or time_embed % 2:
            raise ValueError('Positive dimensions/steps and even time_embed required')
        self.kind, self.steps = kind, int(steps)
        self.action_dims = dict(action_dims)
        self.robot_ids = {r:i for i,r in enumerate(action_dims)}
        self.noise_seed = int(noise_seed)
        self._generators = {}
        # Preserve global learner/environment RNG and all common baseline weights.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(init_seed)
            self.robot_embedding = nn.Embedding(len(action_dims), robot_embed)
            dim = condition_dim + robot_embed
            if kind == 'flow':
                self.action_inputs = nn.ModuleDict({r:nn.Linear(d, action_embed) for r,d in action_dims.items()})
                self.register_buffer('time_frequencies', torch.exp(torch.linspace(0, math.log(1000), time_embed//2)))
                dim += action_embed + time_embed
            self.body = nn.Sequential(nn.Linear(dim, width), nn.ELU(), nn.Linear(width, width), nn.ELU())
            self.outputs = nn.ModuleDict({r:nn.Linear(width, d) for r,d in action_dims.items()})
            for head in self.outputs.values():
                nn.init.orthogonal_(head.weight, gain=.01)
                nn.init.zeros_(head.bias)

    def _condition(self, robot, condition):
        embedding = self.robot_embedding.weight[self.robot_ids[robot]].expand(len(condition), -1)
        return torch.cat((condition, embedding), -1)

    def velocity(self, robot, x, tau, condition):
        if self.kind != 'flow':
            raise ValueError('Velocity exists only for flow decoders')
        angles = tau.reshape(-1, 1)*self.time_frequencies[None, :]
        time = torch.cat((angles.sin(), angles.cos()), -1)
        features = torch.cat((self._condition(robot, condition), self.action_inputs[robot](x), time), -1)
        return self.outputs[robot](self.body(features))

    def predict(self, robot, condition, *, steps=None, initial=None):
        if self.kind == 'mlp':
            return self.outputs[robot](self.body(self._condition(robot, condition)))
        count = self.steps if steps is None else int(steps)
        if count <= 0:
            raise ValueError('Euler steps must be positive')
        shape = (len(condition), self.action_dims[robot])
        if initial is not None and initial.shape != shape:
            raise ValueError('Initial flow action shape differs from target robot')
        x = condition.new_zeros(shape) if initial is None else initial.clone()
        for k in range(count):
            tau = condition.new_full((len(condition),), k/count)
            x = x + self.velocity(robot, x, tau, condition)/count
        return x

    def _generator(self, device):
        key = str(device)
        if key not in self._generators:
            self._generators[key] = torch.Generator(device=device).manual_seed(self.noise_seed)
        return self._generators[key]

    def matching_loss(self, robot, condition, target, *, noise=None, tau=None):
        if self.kind != 'flow':
            raise ValueError('Matching loss exists only for flow decoders')
        target = target.detach()
        generator = self._generator(target.device)
        if noise is None:
            noise = torch.randn(target.shape, device=target.device, dtype=target.dtype, generator=generator)
        if tau is None:
            tau = torch.rand(len(target), device=target.device, dtype=target.dtype, generator=generator)
        x = (1-tau[:, None])*noise + tau[:, None]*target
        return (self.velocity(robot, x, tau, condition)-(target-noise)).square().mean()

    def rng_state(self):
        return {key:generator.get_state().cpu() for key,generator in self._generators.items()}

    def restore_rng(self, states):
        self._generators = {}
        for key, value in states.items():
            self._generator(torch.device(key)).set_state(value.cpu())
