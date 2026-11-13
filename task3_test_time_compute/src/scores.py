"""Rollout quality signals: mean log-prob, self-certainty, visual engagement, VisConf (+ token fusion)."""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np

EPS_V = 1e-12  # visual-engagement denominator guard
EPS_C = 1e-8   # calibration range guard


def visual_engagement_tokens(img, prm, gen) -> np.ndarray:
    """v_t = A_img / (A_img + A_prompt + A_gen); normalised (totals need not sum to one)."""
    img = np.asarray(img, dtype=np.float64)
    prm = np.asarray(prm, dtype=np.float64)
    gen = np.asarray(gen, dtype=np.float64)
    return img / np.maximum(img + prm + gen, EPS_V)


def rollout_scalars(logp, kl, v_tokens: Optional[np.ndarray] = None) -> Dict[str, float]:
    """L_i, K_i (and V_i). Zero-length rollouts give NaN (mapped to -inf when ranking)."""
    T = len(logp)
    if T == 0:
        return {"L": np.nan, "K": np.nan, "V": np.nan}
    out = {"L": float(np.mean(np.asarray(logp, dtype=np.float64))), "K": float(np.mean(np.asarray(kl, dtype=np.float64)))}
    out["V"] = float(np.mean(v_tokens)) if v_tokens is not None else np.nan
    return out


def calibration_bounds(pooled: np.ndarray):
    """5th / 95th percentile with linear interpolation."""
    q05, q95 = np.percentile(pooled, [5, 95], method="linear")
    return float(q05), float(q95)


def calibrate(k: np.ndarray, q05: float, q95: float) -> np.ndarray:
    """c_t = clip_[0,1]((k_t - q05) / max(q95 - q05, 1e-8))."""
    return np.clip((np.asarray(k, dtype=np.float64) - q05) / max(q95 - q05, EPS_C), 0.0, 1.0)


def prefix_visconf(tok_k: Sequence[np.ndarray], tok_v: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
    """VisConf over a candidate prefix. Calibration pools token KL values of THESE candidates only.

    Returns arrays over the candidates: C (mean calibrated confidence), V (mean visual engagement),
    Z = V*C, Zt = mean_t(c_t v_t) (token fusion), cov = Zt - Z (= population Cov_t(c, v))."""
    n = len(tok_k)
    C = np.full(n, np.nan)
    V = np.full(n, np.nan)
    Z = np.full(n, np.nan)
    Zt = np.full(n, np.nan)
    nonempty = [np.asarray(k, dtype=np.float64) for k in tok_k if len(k)]
    if not nonempty:
        return {"C": C, "V": V, "Z": Z, "Zt": Zt, "cov": Zt - Z}
    q05, q95 = calibration_bounds(np.concatenate(nonempty))
    for i in range(n):
        if len(tok_k[i]) == 0:
            continue
        c = calibrate(tok_k[i], q05, q95)
        v = np.asarray(tok_v[i], dtype=np.float64)
        C[i] = c.mean()
        V[i] = v.mean()
        Z[i] = V[i] * C[i]
        Zt[i] = float(np.mean(c * v))
    return {"C": C, "V": V, "Z": Z, "Zt": Zt, "cov": Zt - Z}


def neginf(x: np.ndarray) -> np.ndarray:
    """Map NaN (undefined score for empty rollouts) to -inf so they rank last."""
    x = np.asarray(x, dtype=np.float64)
    return np.where(np.isfinite(x), x, -np.inf)


ALL_SCORES = ("logp", "selfcert", "shortest", "vision", "visconf", "tokenfusion")


def prefix_quality(qd, idx: np.ndarray, names: Iterable[str] = ALL_SCORES) -> Dict[str, np.ndarray]:
    """Quality scores (higher = better) for the candidates `idx` (a prefix of a rollout order).

    'logp' mean sampled-token log-prob, 'selfcert' self-certainty K, 'shortest' = -length (length-only
    control), 'vision' V, 'visconf' Z = V*C (candidate-calibrated), 'tokenfusion' Z^token."""
    names = set(names)
    q: Dict[str, np.ndarray] = {}
    if "logp" in names:
        q["logp"] = neginf(qd.L[idx])
    if "selfcert" in names:
        q["selfcert"] = neginf(qd.K[idx])
    if "shortest" in names:
        q["shortest"] = -qd.T[idx].astype(np.float64)
    if names & {"vision", "visconf", "tokenfusion"} and qd.tok_k is not None:
        if "vision" in names:
            q["vision"] = neginf(qd.V[idx])
        if names & {"visconf", "tokenfusion"}:
            vc = prefix_visconf([qd.tok_k[i] for i in idx], [qd.tok_v[i] for i in idx])
            if "visconf" in names:
                q["visconf"] = neginf(vc["Z"])
            if "tokenfusion" in names:
                q["tokenfusion"] = neginf(vc["Zt"])
    return q
