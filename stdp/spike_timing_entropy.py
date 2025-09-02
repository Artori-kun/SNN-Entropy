
# spike_timing_entropy.py
# Utilities to compute timing-based accuracies and entropy metrics for SNN outputs.
import numpy as np

EPS = 1e-12

# ---------- Readouts (timing-based) ----------

def predict_first_spike(out_ids, out_t_rel, Np):
    """Return 0/1 by comparing first spike time of 2 populations."""
    if out_ids is None or len(out_ids)==0:
        return 0
    out_ids = np.asarray(out_ids); out_t_rel = np.asarray(out_t_rel, dtype=float)
    has0 = np.any(out_ids < Np); has1 = np.any(out_ids >= Np)
    t0 = out_t_rel[out_ids < Np].min() if has0 else np.inf
    t1 = out_t_rel[out_ids >= Np].min() if has1 else np.inf
    if np.isinf(t0) and np.isinf(t1): return 0
    return 0 if t0 <= t1 else 1

def predict_weighted(out_ids, out_t_rel, Np, tau=30.0):
    """Time-weighted vote: earlier spikes count more (exp decay with tau in ms)."""
    if out_ids is None or len(out_ids)==0:
        return 0
    out_ids = np.asarray(out_ids); out_t_rel = np.asarray(out_t_rel, dtype=float)
    s0 = np.exp(-out_t_rel[out_ids <  Np]/float(tau)).sum()
    s1 = np.exp(-out_t_rel[out_ids >= Np]/float(tau)).sum()
    return 0 if s0 >= s1 else 1

def predict_race_to_N(out_ids, out_t_rel, Np, N=3):
    """Race to N spikes: the first population that accumulates N spikes wins."""
    if out_ids is None or len(out_ids)==0:
        return 0
    out_ids = np.asarray(out_ids); out_t_rel = np.asarray(out_t_rel, dtype=float)
    order = np.argsort(out_t_rel)
    c0=c1=0
    for k in order:
        if out_ids[k] < Np: c0 += 1
        else: c1 += 1
        if c0>=N or c1>=N:
            return 0 if c0>c1 else 1
    # fallback: majority
    return 0 if (out_ids < Np).sum() >= (out_ids >= Np).sum() else 1

def predict_count(out_ids, Np):
    """Simple spike count vote."""
    if out_ids is None or len(out_ids)==0:
        return 0
    out_ids = np.asarray(out_ids)
    return 0 if (out_ids < Np).sum() >= (out_ids >= Np).sum() else 1

# ---------- Accuracy helpers ----------

def accuracy(events, labels, Np, method="weighted", **kwargs):
    """
    events: list of (out_ids, out_t_rel) for each sample
    labels: array-like of 0/1
    method: "first", "weighted", "raceN", "count"
    kwargs: tau (for weighted), N (for raceN)
    """
    labels = np.asarray(labels, dtype=int)
    preds = []
    for (ids, t) in events:
        if method == "first":
            yh = predict_first_spike(ids, t, Np)
        elif method == "weighted":
            yh = predict_weighted(ids, t, Np, tau=kwargs.get("tau", 30.0))
        elif method == "raceN":
            yh = predict_race_to_N(ids, t, Np, N=kwargs.get("N", 3))
        elif method == "count":
            yh = predict_count(ids, Np)
        else:
            raise ValueError("Unknown method")
        preds.append(yh)
    preds = np.asarray(preds, dtype=int)
    return float((preds == labels).mean()), preds

# ---------- Spike-train reconstruction ----------

def out_spike_train(out_ids, out_t_rel, Nout, T, dt_ms):
    """
    Build (Nout, T) binary matrix for outputs from ids & times (ms).
    Multiple spikes in the same bin are clipped to 1.
    """
    S = np.zeros((Nout, T), dtype=np.uint8)
    if out_ids is None or len(out_ids)==0: return S
    out_ids = np.asarray(out_ids, dtype=int)
    out_t_rel = np.asarray(out_t_rel, dtype=float)
    bins = np.clip((out_t_rel / float(dt_ms)).astype(int), 0, T-1)
    S[out_ids, bins] = 1
    return S

# ---------- Entropy over spike times (histogram over time) ----------

def shannon_entropy_hist(times_ms, T, dt_ms):
    """
    Entropy (bits) of spike-time distribution across T bins of width dt_ms.
    If no spikes -> 0.
    """
    if times_ms is None or len(times_ms)==0:
        return 0.0
    bins = np.arange(0, T*dt_ms + 1e-6, dt_ms)
    hist, _ = np.histogram(times_ms, bins=bins)
    total = hist.sum()
    if total == 0: return 0.0
    p = hist.astype(np.float64) / total
    p = p[p>0]
    return float(-(p*np.log2(p)).sum())

def group_time_entropy(out_ids, out_t_rel, Np, T, dt_ms):
    """
    Return entropy of time hist for each population and combined.
    """
    ids = np.asarray(out_ids) if out_ids is not None else np.array([], dtype=int)
    tms = np.asarray(out_t_rel, dtype=float) if out_t_rel is not None else np.array([], dtype=float)
    e0 = shannon_entropy_hist(tms[ids <  Np], T, dt_ms)
    e1 = shannon_entropy_hist(tms[ids >= Np], T, dt_ms)
    e_all = shannon_entropy_hist(tms, T, dt_ms)
    return e0, e1, e_all

# ---------- Entropy of binary occupancy (rate-sensitive) ----------

def shannon_entropy_binary(p):
    """Binary entropy H(p) in bits."""
    if p <= 0.0 or p >= 1.0: return 0.0
    return float(-(p*np.log2(p) + (1-p)*np.log2(1-p)))

def occupancy_entropy(S):
    """
    S: (n, T) binary matrix. Returns H over Bernoulli with p = mean occupancy.
    """
    p = float(S.mean())
    return shannon_entropy_binary(p)

# ---------- Margin-based metrics ----------

def margin_weighted(out_ids, out_t_rel, Np, tau=30.0):
    """Return M = S1 - S0 (weighted sums)."""
    if out_ids is None or len(out_ids)==0: return 0.0
    out_ids = np.asarray(out_ids); out_t_rel = np.asarray(out_t_rel, dtype=float)
    s0 = np.exp(-out_t_rel[out_ids <  Np]/float(tau)).sum()
    s1 = np.exp(-out_t_rel[out_ids >= Np]/float(tau)).sum()
    return float(s1 - s0)

def entropy_from_continuous(x, bins="fd"):
    """Estimate H(x) with histogram (Freedman-Diaconis by default)."""
    x = np.asarray(x, dtype=float)
    if x.size == 0: return 0.0, np.array([])
    if isinstance(bins, str) and bins == "fd":
        q75, q25 = np.percentile(x, [75, 25])
        iqr = q75 - q25
        if iqr <= EPS:
            bins = 20
        else:
            bw = 2*iqr * (x.size ** (-1/3))
            if bw <= EPS:
                bins = 20
            else:
                rng = x.max() - x.min() + EPS
                bins = max(10, int(np.ceil(rng / bw)))
    hist, edges = np.histogram(x, bins=bins)
    p = hist.astype(np.float64) / max(1, hist.sum())
    p = p[p>0]
    H = float(-(p*np.log2(p)).sum())
    return H, edges

def margin_entropy(events, labels, Np, tau=30.0, bins="fd"):
    """
    events: list of (out_ids, out_t_rel)
    labels: 0/1
    Returns dict with H(M), H(M|0), H(M|1), MI.
    """
    labels = np.asarray(labels, dtype=int)
    Ms = np.array([margin_weighted(ids, t, Np, tau=tau) for (ids,t) in events], dtype=float)
    H_all, edges = entropy_from_continuous(Ms, bins=bins)
    M0 = Ms[labels==0]; M1 = Ms[labels==1]
    H0,_ = entropy_from_continuous(M0, bins=edges)
    H1,_ = entropy_from_continuous(M1, bins=edges)
    p1 = float((labels==1).mean())
    p0 = 1.0 - p1
    H_cond = p0*H0 + p1*H1
    MI = H_all - H_cond
    return {"H_M": H_all, "H_M_given0": H0, "H_M_given1": H1, "H_M_cond": H_cond, "MI": MI,
            "M_all": Ms, "bins": edges}

# ---------- Convenience to compute all metrics on a dataset ----------

def evaluate_timing(events, labels, Np, tau=30.0, N=3):
    acc_w, _ = accuracy(events, labels, Np, method="weighted", tau=tau)
    acc_f, _ = accuracy(events, labels, Np, method="first")
    acc_r, _ = accuracy(events, labels, Np, method="raceN", N=N)
    acc_c, _ = accuracy(events, labels, Np, method="count")
    return {"acc_weighted": acc_w, "acc_first": acc_f, "acc_raceN": acc_r, "acc_count": acc_c}

def evaluate_entropy_time(events, Np, T, dt_ms):
    e0_list=[]; e1_list=[]; eall_list=[]
    for ids, t in events:
        e0,e1,eall = group_time_entropy(ids, t, Np, T, dt_ms)
        e0_list.append(e0); e1_list.append(e1); eall_list.append(eall)
    return {"H_time_group0_mean": float(np.mean(e0_list)),
            "H_time_group1_mean": float(np.mean(e1_list)),
            "H_time_model_mean": float(np.mean(eall_list)),
            "per_sample": {"g0": e0_list, "g1": e1_list, "all": eall_list}}
