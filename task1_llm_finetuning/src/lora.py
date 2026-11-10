"""LoRA adapters (own implementation, so LoRA-4/16 and OPLoRA share *exactly* the same code path & init).

W = W0 + (alpha/r) * B_eff @ A_eff
  LoRA   : A_eff = A, B_eff = B
  OPLoRA : B_eff = B - Uk(Uk^T B), A_eff = A - (A Vk)Vk^T   (bases frozen buffers, computed once in FP32)
A: (r, d_in) kaiming-uniform(a=sqrt(5)) ; B: (d_out, r) zeros ; adapters FP32 ; backbone frozen (FP16 on GPU).
"""
import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        assert base.bias is None, "no trainable biases; targets are bias-free in Qwen3"
        self.base = base
        self.r, self.alpha, self.scale = r, alpha, alpha / r
        self.in_features, self.out_features = base.in_features, base.out_features
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features, dtype=torch.float32))  # CPU init (device-independent)
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("Uk", None, persistent=False)  # (d_out, k) FP32
        self.register_buffer("Vk", None, persistent=False)  # (d_in, k)  FP32
        self.w0_fro = None

    def set_bases(self, Uk: torch.Tensor, Vk: torch.Tensor):
        self.Uk, self.Vk = Uk, Vk

    def effective_factors(self):
        """FP32 effective factors (autocast disabled so projections are exact FP32)."""
        with torch.autocast(device_type=self.lora_A.device.type, enabled=False):
            A, B = self.lora_A.float(), self.lora_B.float()
            if self.Uk is not None:
                B = B - self.Uk @ (self.Uk.transpose(0, 1) @ B)
                A = A - (A @ self.Vk) @ self.Vk.transpose(0, 1)
        return A, B

    def forward(self, x):
        out = self.base(x)
        A, B = self.effective_factors()
        z = F.linear(self.dropout(x), A.to(x.dtype))
        z = F.linear(z, B.to(x.dtype))
        return out + (z * self.scale).to(out.dtype)

    @torch.no_grad()
    def delta_weight(self) -> torch.Tensor:
        A, B = self.effective_factors()
        return self.scale * (B @ A)  # (d_out, d_in) FP32


def inject_lora(model, targets: List[str], r: int, alpha: float, dropout: float, seed: int) -> Dict[str, LoRALinear]:
    """Freeze backbone, wrap target Linears in every transformer layer. Deterministic init (seeded just before)."""
    for p in model.parameters():
        p.requires_grad = False
    torch.manual_seed(seed)  # identical A init for LoRA-16 and OPLoRA-16
    modules: Dict[str, LoRALinear] = {}
    named = [(n, m) for n, m in model.model.layers.named_modules()]
    for name, mod in named:
        for child_name, child in list(mod.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                wrapped = LoRALinear(child, r, alpha, dropout)
                setattr(mod, child_name, wrapped)
                full = f"model.layers.{name}.{child_name}" if name else f"model.layers.{child_name}"
                modules[full] = wrapped
    device = next(model.parameters()).device
    for m in modules.values():
        m.lora_A.data = m.lora_A.data.to(device)
        m.lora_B.data = m.lora_B.data.to(device)
        m.w0_fro = m.base.weight.detach().float().norm().item()
    return modules


def adapter_parameters(modules: Dict[str, LoRALinear]):
    return [p for m in modules.values() for p in (m.lora_A, m.lora_B)]


def adapter_state(modules: Dict[str, LoRALinear]) -> Dict[str, torch.Tensor]:
    st = {}
    for n, m in modules.items():
        st[f"{n}.lora_A"] = m.lora_A.detach().cpu().contiguous()
        st[f"{n}.lora_B"] = m.lora_B.detach().cpu().contiguous()
    return st


def load_adapter_state(modules: Dict[str, LoRALinear], state: Dict[str, torch.Tensor]):
    for n, m in modules.items():
        m.lora_A.data.copy_(state[f"{n}.lora_A"].to(m.lora_A.device))
        m.lora_B.data.copy_(state[f"{n}.lora_B"].to(m.lora_B.device))


def init_fingerprint(modules: Dict[str, LoRALinear]) -> float:
    """Sum of |A| at init: compare LoRA-16 vs OPLoRA-16 logs to confirm identical initial factors."""
    return float(sum(m.lora_A.detach().double().abs().sum().item() for m in modules.values()))


@torch.no_grad()
def adapter_delta_rows(modules: Dict[str, LoRALinear]) -> List[dict]:
    rows = []
    for n, m in modules.items():
        dW = m.delta_weight()
        d = dW.norm().item()
        rows.append({
            "module": n, "shape": f"{m.out_features}x{m.in_features}", "rank": m.r,
            "A_fro": m.lora_A.norm().item(), "B_fro": m.lora_B.norm().item(),
            "delta_fro": d, "w0_fro": m.w0_fro, "eta": d / m.w0_fro,
        })
    return rows
