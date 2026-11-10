"""L2-SP: Omega(theta, theta0) = 0.5 * sum_j (theta_j - theta0_j)^2 over ALL unique parameters
(biases/norms included, tied parameters counted once = model.named_parameters() de-duplication).
theta0 is a fixed FP32 copy. Omega is a SUM (not mean). Penalty gradient lam*(theta-theta0) is added once per
optimizer update (the SFT loss is the effective-batch token mean), scaled by the GradScaler scale so it is
correct after unscale_.
Displacement identity: ||theta - theta0||_2 = sqrt(2 * Omega).
"""
import time
from typing import List, Tuple

import torch


class Reference:
    """Fixed starting-point copy theta0 (FP32). Used by L2-SP (on GPU) and by Full-FT displacement logging (CPU)."""

    def __init__(self, model, device, pin: bool = False):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        self.names: List[str] = []
        self.params: List[torch.nn.Parameter] = []
        self.ref: List[torch.Tensor] = []
        for n, p in model.named_parameters():  # de-duplicates tied weights
            self.names.append(n)
            self.params.append(p)
            r = p.detach().clone().float().to(device)
            self.ref.append(r.pin_memory() if (pin and torch.cuda.is_available() and r.device.type == 'cpu') else r)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.transfer_seconds = time.perf_counter() - t0
        self.storage_bytes = sum(r.numel() * r.element_size() for r in self.ref)
        self.device = device

    @torch.no_grad()
    def displacement_rows(self) -> Tuple[List[dict], float]:
        rows, tot = [], 0.0
        for n, p, r in zip(self.names, self.params, self.ref):
            d = (p.detach().float() - r.to(p.device))
            d2 = (d * d).sum(dtype=torch.float64).item()
            w2 = (r.to(p.device) ** 2).sum(dtype=torch.float64).item()
            tot += d2
            rows.append({"name": n, "numel": p.numel(), "shape": "x".join(map(str, p.shape)),
                         "delta_fro": d2 ** 0.5, "w0_fro": w2 ** 0.5, "eta": (d2 ** 0.5) / max(w2 ** 0.5, 1e-12)})
        return rows, tot ** 0.5


class L2SP(Reference):
    def __init__(self, model, lam: float, device, pin: bool = True):
        # device = where theta0 lives ('cuda' = fast, +2.4GB GPU; 'cpu' = pinned host copy, streamed per update)
        super().__init__(model, device, pin=pin)
        self.lam = lam
        self.penalty_seconds_total = 0.0

    @torch.no_grad()
    def add_penalty_grads(self, grad_scale: float = 1.0) -> float:
        """Adds lam*(p-p0)*grad_scale to p.grad; returns Omega (float, FP64-accumulated, pre-step params)."""
        use_cuda = self.params[0].is_cuda
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        omega = torch.zeros((), dtype=torch.float64, device=self.params[0].device)
        for p, r in zip(self.params, self.ref):
            d = p.detach() - r.to(p.device, non_blocking=True)
            omega += 0.5 * (d * d).sum(dtype=torch.float64)
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            p.grad.add_(d, alpha=self.lam * grad_scale)
        val = omega.item()
        if use_cuda:
            torch.cuda.synchronize()
        self.penalty_seconds_total += time.perf_counter() - t0
        self.last_penalty_seconds = time.perf_counter() - t0
        return val
