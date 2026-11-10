import math
import torch
from task1_llm_finetuning.src.l2sp import L2SP
from task1_llm_finetuning.tests.tiny import tiny_model


def test_omega_grad_and_displacement_identity():
    m = tiny_model()
    l2 = L2SP(m, lam=1e-3, device=torch.device("cpu"))
    # tied weights counted once
    assert len(l2.params) == len(list(m.named_parameters()))
    assert sum(p.numel() for p in l2.params) == sum(p.numel() for p in m.parameters())
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    for p in m.parameters():
        p.grad = torch.zeros_like(p)
    omega = l2.add_penalty_grads(grad_scale=4.0)
    exp = sum(0.5 * ((p.detach() - r) ** 2).sum().item() for p, r in zip(l2.params, l2.ref))
    assert math.isclose(omega, exp, rel_tol=1e-6)
    for p, r in zip(l2.params, l2.ref):
        assert torch.allclose(p.grad, 1e-3 * 4.0 * (p.detach() - r), atol=1e-8)
    _, disp = l2.displacement_rows()
    assert math.isclose(disp, math.sqrt(2 * omega), rel_tol=1e-6)
