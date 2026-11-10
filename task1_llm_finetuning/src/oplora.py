"""OPLoRA: SVD bases (computed once, FP32), numerical constraint verification, projected-factor export."""
import time
from typing import Dict, List, Tuple

import torch

from .lora import LoRALinear


@torch.no_grad()
def compute_bases(modules: Dict[str, LoRALinear], k: int, svd_device: str, target_device) -> Tuple[List[dict], float]:
    """Leading-k singular bases of each frozen W0 (FP32). Returns (per-module meta rows, total seconds)."""
    rows, t_total = [], 0.0
    for name, m in modules.items():
        t0 = time.perf_counter()
        W = m.base.weight.detach().float().to(svd_device)
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        Uk = U[:, :k].contiguous().to(target_device)
        Vk = Vh[:k].transpose(0, 1).contiguous().to(target_device)
        m.set_bases(Uk, Vk)
        dt = time.perf_counter() - t0
        t_total += dt
        S2 = S.double().pow(2)
        rows.append({
            "module": name, "shape": f"{W.shape[0]}x{W.shape[1]}", "k": k,
            "sigma_1": S[0].item(), "sigma_k": S[k - 1].item(),
            "energy_frac_topk": (S2[:k].sum() / S2.sum()).item(), "svd_seconds": dt,
        })
    return rows, t_total


def bases_state(modules: Dict[str, LoRALinear]) -> Dict[str, torch.Tensor]:
    st = {}
    for n, m in modules.items():
        st[f"{n}.Uk"] = m.Uk.detach().cpu().contiguous()
        st[f"{n}.Vk"] = m.Vk.detach().cpu().contiguous()
    return st


@torch.no_grad()
def verify_constraints(modules: Dict[str, LoRALinear]) -> List[dict]:
    """Checks (numerically): Uk^T B~ = 0, A~ Vk = 0, hence Uk^T dW = 0 and dW Vk = 0.
    rho_out = ||Uk Uk^T dW||_F^2/||dW||_F^2 ; rho_in = ||dW Vk Vk^T||_F^2/||dW||_F^2 (both ~0 for OPLoRA)."""
    rows = []
    for n, m in modules.items():
        A, B = m.effective_factors()
        dW = m.scale * (B @ A)
        dn2 = dW.pow(2).sum().item()
        UtB = (m.Uk.T @ B).abs().max().item()
        AV = (A @ m.Vk).abs().max().item()
        rho_out = (m.Uk.T @ dW).pow(2).sum().item() / dn2 if dn2 > 0 else float("nan")
        rho_in = (dW @ m.Vk).pow(2).sum().item() / dn2 if dn2 > 0 else float("nan")
        rows.append({"module": n, "max_abs_UkT_Btilde": UtB, "max_abs_Atilde_Vk": AV,
                     "B_tilde_max_abs": B.abs().max().item(), "A_tilde_max_abs": A.abs().max().item(),
                     "rho_out": rho_out, "rho_in": rho_in, "delta_fro": dW.norm().item()})
    return rows


@torch.no_grad()
def export_projected(modules: Dict[str, LoRALinear]) -> Dict[str, torch.Tensor]:
    """Effective (projected) factors for merging/inference: keys '{module}.A_eff' / '{module}.B_eff'."""
    st = {}
    for n, m in modules.items():
        A, B = m.effective_factors()
        st[f"{n}.A_eff"] = A.cpu().contiguous()
        st[f"{n}.B_eff"] = B.cpu().contiguous()
    return st
