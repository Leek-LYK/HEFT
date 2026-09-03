from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


DEFAULT_STUDENT_LATENT_QUANTIZER_SCALE = 4.499549865722656


@dataclass
class StudentLatentQuantizerCfg:
    """Configuration for the optional Student latent quantizer."""

    enabled: bool = False
    scale: float = DEFAULT_STUDENT_LATENT_QUANTIZER_SCALE


class StudentLatentQuantizer(nn.Module):
    """Apply the deployed tanh/round latent transform with STE gradients."""

    def __init__(
        self,
        enabled: bool = False,
        scale: float = DEFAULT_STUDENT_LATENT_QUANTIZER_SCALE,
    ) -> None:
        super().__init__()
        scale = float(scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"Student latent quantizer scale must be finite and > 0, got {scale}")
        self.enabled = bool(enabled)
        self.scale = scale

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Return the unchanged latent or its straight-through quantization."""
        if not self.enabled:
            return latent
        continuous = torch.tanh(latent) * self.scale
        return continuous + (torch.round(continuous) - continuous).detach()
