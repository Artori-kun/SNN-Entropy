#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ensemble readouts for 2-class population outputs (no hidden layer required).

Combines 4 timing/count readouts:
- count margin
- time-weighted margin (exp decay, tau)
- first-spike latency margin
- race-to-N margin (time to reach N spikes)

Provides:
- per-sample margin extraction
- simple calibration of global weights from a calibration set (train/val)
- ensemble prediction using weighted sum of (normalized) margins
- optional per-sample confidence weighting

All numpy; no sklearn dependency.
"""
import numpy as np

EPS = 1e-9

# ---------- Low-level utilities ----------

def _first_times(ids, t, Np):
    ids = np.asarray(ids, dtype=int)
    t = np.asarray(t, dtype=float)
    has0 = np.any(ids < Np); has1 = np.any(ids >= Np)
    t0 = t[ids < Np].min() if has0 else np.inf
    t1 = t[ids >= Np].min() if has1 else np.inf
    return t0, t1

def _weighted_scores(ids, t, Np, tau):
    ids = np.asarray(ids, dtype=int)
    t = np.asarray(t, dtype=float)
    s0 = np.exp(-t[ids <  Np]/float(tau)).sum()
    s1 = np.exp(-t[ids >= Np]/float(tau)).sum()
    return s0, s1

def _counts(ids, Np):
    ids = np.asarray(ids, dtype=int)
    return int((ids < Np).sum()), int((ids >= Np).sum())

def _time_to_N(ids, t, Np, N):
    # Return time at which each population accumulates N spikes (np.inf if never)
    ids = np.asarray(ids, dtype=int)
    t = np.asarray(t, dtype=float)
    if ids.size == 0: return np.inf, np.inf
    order = np.argsort(t)
    c0 = c1 = 0
    tN0 = np.inf; tN1 = np.inf
    for k in order:
        if ids[k] < Np:
            c0 += 1
            if c0 == N and np.isinf(tN0): tN0 = t[k]
        else:
            c1 += 1
            if c1 == N and np.isinf(tN1): tN1 = t[k]
    return tN0, tN1

# ---------- Per-sample margins (positive -> favor class 1) ----------

def compute_margins(ids, t, Np, T, tau=30.0, N=3):
    """
    ids: array of output neuron indices
    t:   times (ms) aligned to sample start
    Returns dict with margins in [-1,1]:
      m_count, m_weight, m_first, m_race
    Also returns 'conf' dictionary with per-sample confidence proxies.
    """
    ids = np.asarray(ids, dtype=int); t = np.asarray(t, dtype=float)
    # Count margin in [-1,1]
    c0, c1 = _counts(ids, Np)
    total = c0 + c1
    m_count = 0.0 if total == 0 else (c1 - c0) / float(total)

    # Weighted margin
    s0, s1 = _weighted_scores(ids, t, Np, tau)
    denom = (s1 + s0)
    m_weight = 0.0 if denom <= EPS else (s1 - s0) / (denom + EPS)

    # First-spike latency difference (normalize by T)
    t0, t1 = _first_times(ids, t, Np)
    if np.isinf(t0) and np.isinf(t1):
        m_first = 0.0
        gap_first = 0.0
    else:
        if np.isinf(t0) and not np.isinf(t1):   # only class1 spiked
            m_first = +1.0
        elif not np.isinf(t0) and np.isinf(t1): # only class0 spiked
            m_first = -1.0
        else:
            gap = (t0 - t1) / float(T)  # >0 means class1 earlier
            m_first = float(np.clip(-gap, -1.0, 1.0))  # -gap so that positive favors class1
        gap_first = 0.0 if (np.isinf(t0) or np.isinf(t1)) else abs((t0 - t1)/float(T))

    # Race-to-N: time to reach N spikes
    tN0, tN1 = _time_to_N(ids, t, Np, N)
    if np.isinf(tN0) and np.isinf(tN1):
        m_race = 0.0
        gap_race = 0.0
    else:
        if np.isinf(tN0) and not np.isinf(tN1):
            m_race = +1.0
        elif not np.isinf(tN0) and np.isinf(tN1):
            m_race = -1.0
        else:
            gapN = (tN0 - tN1) / float(T)
            m_race = float(np.clip(-gapN, -1.0, 1.0))  # positive favors class1
        gap_race = 0.0 if (np.isinf(tN0) or np.isinf(tN1)) else abs((tN0 - tN1)/float(T))

    conf = {
        "count_total": total,
        "gap_first": gap_first,   # 0..1
        "gap_race":  gap_race,    # 0..1
        "s_sum": s0 + s1,
    }
    return {
        "m_count": m_count,
        "m_weight": m_weight,
        "m_first": m_first,
        "m_race": m_race,
        "conf": conf
    }

# ---------- Calibration of global weights from a labeled set ----------

def calibrate_weights(events, labels, Np, T, tau=30.0, N=3, gamma=1.0, use_std_norm=True):
    """
    Learn simple global weights from a calibration split (e.g., training or a CV fold).
    Strategy:
      1) Compute method accuracies vs labels -> a_k
      2) Set w_k ∝ max(a_k - 0.5, 0)^gamma (zero out near chance)
      3) Normalise weights to sum 1 (fallback to equal if all zero)
      4) Optionally z-normalise margins by their std across the calibration set
    Returns dict: {'w': {'count':..., 'weight':..., 'first':..., 'race':...},
                   'scale': {'count': s_count, ...}, 'acc_calib': {...} }
    """
    labels = np.asarray(labels, dtype=int)
    Ms = {"count": [], "weight": [], "first": [], "race": []}
    preds = {"count": [], "weight": [], "first": [], "race": []}

    for (ids, t), y in zip(events, labels):
        m = compute_margins(ids, t, Np, T, tau=tau, N=N)
        Ms["count"].append(m["m_count"])
        Ms["weight"].append(m["m_weight"])
        Ms["first"].append(m["m_first"])
        Ms["race"].append(m["m_race"])
        preds["count"].append(1 if m["m_count"]>0 else 0)
        preds["weight"].append(1 if m["m_weight"]>0 else 0)
        preds["first"].append(1 if m["m_first"]>0 else 0)
        preds["race"].append(1 if m["m_race"]>0 else 0)

    acc = {k: float((np.array(v)==labels).mean()) for k, v in preds.items()}
    # weights from accuracy above chance
    raw = {k: max(acc[k]-0.5, 0.0)**gamma for k in acc}
    s = sum(raw.values())
    if s <= EPS:
        w = {k: 0.25 for k in raw}  # equal weights
    else:
        w = {k: raw[k]/s for k in raw}

    # scales for z-normalisation (std)
    scale = {}
    if use_std_norm:
        for k, arr in Ms.items():
            std = float(np.std(arr)) + EPS
            scale[k] = std
    else:
        scale = {k: 1.0 for k in Ms}

    out = {"w": w, "scale": scale, "acc_calib": acc}
    return out

# ---------- Ensemble prediction ----------

def ensemble_predict(ids, t, Np, T, calib, tau=30.0, N=3, per_sample_conf=True):
    """
    Combine normalised margins with global weights and (optional) per-sample confidence.
    Returns (y_hat, details)
    """
    m = compute_margins(ids, t, Np, T, tau=tau, N=N)
    # normalise
    m_norm = {
        "count": m["m_count"] / calib["scale"]["count"],
        "weight": m["m_weight"] / calib["scale"]["weight"],
        "first": m["m_first"] / calib["scale"]["first"],
        "race": m["m_race"] / calib["scale"]["race"],
    }
    # optional confidence (0..1 proxies)
    if per_sample_conf:
        conf = m["conf"]
        # scale gaps already in [0,1]; for counts, use a saturating mapping
        c_weight = min(1.0, conf["s_sum"] / (conf["s_sum"] + 5.0))
        c_count  = min(1.0, conf["count_total"] / (conf["count_total"] + 5.0))
        c_first  = conf["gap_first"]
        c_race   = conf["gap_race"]
    else:
        c_weight = c_count = c_first = c_race = 1.0

    # weighted sum
    w = calib["w"]
    score = (
        w["count"]  * m_norm["count"]  * c_count +
        w["weight"] * m_norm["weight"] * c_weight +
        w["first"]  * m_norm["first"]  * c_first +
        w["race"]   * m_norm["race"]   * c_race
    )
    y_hat = 1 if score > 0 else 0
    details = {"score": float(score), "m": m, "m_norm": m_norm, "conf": m["conf"], "w": w}
    return y_hat, details

def ensemble_accuracy(events, labels, Np, T, calib, tau=30.0, N=3, per_sample_conf=True):
    labels = np.asarray(labels, dtype=int)
    preds = []
    for (ids, t) in events:
        y_hat, _ = ensemble_predict(ids, t, Np, T, calib, tau=tau, N=N, per_sample_conf=per_sample_conf)
        preds.append(y_hat)
    preds = np.asarray(preds, dtype=int)
    return float((preds == labels).mean()), preds

if __name__ == "__main__":
    print("This module provides ensemble readouts; import and use in your notebook.")
