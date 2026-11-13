"""Trainable vision-to-language connector g_phi and its magnitude calibration."""
from __future__ import annotations

import torch
import torch.nn as nn

CALIBRATION_RULE = (
    "Scale the final Linear (weight and bias) by s = mean_text_token_L2_norm / mean_visual_token_L2_norm, "
    "where the text statistic is the mean L2 norm of Qwen input embeddings over the content tokens "
    "(question + answer/caption) of TRAINING examples and the visual statistic is the mean per-token L2 "
    "norm of the freshly initialised connector output on cached SigLIP features of TRAINING images."
)


class Connector(nn.Module):
    """Linear(768 -> 1024) -> GELU -> Linear(1024 -> 1024), applied per patch token."""

    def __init__(self, in_dim: int = 768, hidden_dim: int = 1024, out_dim: int = 1024) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.in_dim, self.out_dim = in_dim, out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[-1] != self.in_dim:
            raise ValueError(f"expected [B, N, {self.in_dim}] features, got {tuple(x.shape)}")
        return self.fc2(self.act(self.fc1(x)))


@torch.no_grad()
def mean_output_norm(connector: Connector, feats: torch.Tensor, batch_size: int = 32) -> float:
    """Mean per-token L2 norm of the connector output (computed in FP32 on the connector's device)."""
    dev = next(connector.parameters()).device
    was_training = connector.training
    connector.eval()
    total, count = 0.0, 0
    for i in range(0, feats.shape[0], batch_size):
        x = feats[i:i + batch_size].to(dev).float()
        y = connector(x)
        total += y.norm(dim=-1).sum().item()
        count += y.shape[0] * y.shape[1]
    connector.train(was_training)
    return total / max(count, 1)


@torch.no_grad()
def calibrate_connector(connector: Connector, train_feats: torch.Tensor, text_mean_norm: float) -> dict:
    """Rescale the last layer so the mean visual-token norm equals the mean text-token norm."""
    if text_mean_norm <= 0:
        raise ValueError("text embedding norm must be positive")
    before = mean_output_norm(connector, train_feats)
    if before <= 0:
        raise ValueError("connector output norm is zero")
    scale = text_mean_norm / before
    connector.fc2.weight.mul_(scale)
    connector.fc2.bias.mul_(scale)
    after = mean_output_norm(connector, train_feats)
    return {
        "rule": CALIBRATION_RULE,
        "target_text_mean_l2_norm": float(text_mean_norm),
        "visual_mean_l2_norm_before": float(before),
        "scale_factor": float(scale),
        "visual_mean_l2_norm_after": float(after),
        "n_train_images": int(train_feats.shape[0]),
    }
