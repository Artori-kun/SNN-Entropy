#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# STDP (teacher-aided) classifier for NPZ spike datasets (one-folder) + weight logging per epoch
#
# Usage example:
# python stdp_onefolder_wlog.py --npz-dir /path/to/ALL_NPZ --x-key spike_matrix --label-key label \
#   --val-split 0.2 --seed 42 --epochs 15 --epoch-samples 128 --out-pop 8 \
#   --balance-train --balance-val --save weights_log.npz
#
# Requires: brian2

import argparse, glob, json
from pathlib import Path
import numpy as np

from brian2 import (ms, defaultclock, start_scope,
                    NeuronGroup, SpikeGeneratorGroup, Synapses,
                    SpikeMonitor, Network)

EPS = 1e-9

# ---------------------------- Readouts ----------------------------
def predict_first(out_ids, out_t, Np):
    if out_ids.size == 0: return 0
    t0 = out_t[out_ids < Np].min() if np.any(out_ids < Np) else np.inf
    t1 = out_t[out_ids >= Np].min() if np.any(out_ids >= Np) else np.inf
    if np.isinf(t0) and np.isinf(t1): return 0
    return 0 if t0 <= t1 else 1

def predict_weighted(out_ids, out_t, Np, tau=30.0):
    if out_ids.size == 0: return 0
    s0 = np.exp(-out_t[out_ids <  Np]/tau).sum()
    s1 = np.exp(-out_t[out_ids >= Np]/tau).sum()
    return 0 if s0 >= s1 else 1

def predict_raceN(out_ids, out_t, Np, N=3):
    if out_ids.size == 0: return 0
    order = np.argsort(out_t)
    c0=c1=0
    for k in order:
        if out_ids[k] < Np: c0+=1
        else: c1+=1
        if c0>=N or c1>=N:
            return 0 if c0>c1 else 1
    return 0 if (out_ids < Np).sum() >= (out_ids >= Np).sum() else 1

def predict_count(out_ids, Np):
    if out_ids.size == 0: return 0
    return 0 if (out_ids < Np).sum() >= (out_ids >= Np).sum() else 1

def eval_all(events, labels, Np, tau=30.0, N=3):
    y = np.asarray(labels, dtype=int)
    acc = {}; preds = {}
    for name in ["count", "weighted", "first", "raceN"]:
        yh = []
        for ids, t in events:
            ids = np.asarray(ids); t = np.asarray(t, dtype=float)
            if name=="count":   p = predict_count(ids, Np)
            elif name=="first": p = predict_first(ids, t, Np)
            elif name=="raceN": p = predict_raceN(ids, t, Np, N=N)
            else:               p = predict_weighted(ids, t, Np, tau=tau)
            yh.append(p)
        yh = np.asarray(yh, dtype=int)
        preds[name] = yh
        acc[name] = float((yh == y).mean())
    ens = (preds["count"] + preds["weighted"] + preds["first"] + preds["raceN"]) >= 2
    acc["ensemble"] = float((ens.astype(int) == y).mean())
    return acc

# ---------------------------- Data ----------------------------
def load_all_npz(dirpath, x_key="x", label_key="label"):
    files = sorted(glob.glob(str(Path(dirpath) / "*.npz")))
    if len(files)==0: raise FileNotFoundError(f"No .npz in {dirpath}")
    X = []; y = []
    for f in files:
        d = np.load(f)
        X.append(d[x_key])
        y.append(int(d[label_key]))
    X = np.array(X, dtype=np.uint8)
    y = np.array(y, dtype=int)
    return X, y

def stratified_split_indices(y, val_split=0.2, test_split=0.0, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    idx = np.arange(len(y))
    idx_train, idx_val, idx_test = [], [], []
    for c in np.unique(y):
        idc = idx[y==c]
        rng.shuffle(idc)
        n = len(idc)
        n_test = int(round(n*test_split))
        n_val  = int(round(n*val_split))
        n_train = n - n_val - n_test
        idx_test.extend(idc[:n_test])
        idx_val.extend(idc[n_test:n_test+n_val])
        idx_train.extend(idc[n_test+n_val:])
    rng.shuffle(idx_train); rng.shuffle(idx_val); rng.shuffle(idx_test)
    return np.array(idx_train), np.array(idx_val), np.array(idx_test)

def make_spike_indices(X, dt_ms=1.0):
    C, T = X.shape
    ch, tbin = np.where(X > 0)
    times = (tbin.astype(float) * dt_ms)
    return ch.astype(int), times

# ---------------------------- Network ----------------------------
def build_net(C, Np, params):
    start_scope()
    defaultclock.dt = params["dt_ms"]*ms

    G_in = SpikeGeneratorGroup(C, indices=np.array([], dtype=int), times=np.array([])*ms)

    eqs = '''
    dv/dt = (-(v - v_rest) + ge - gi + I_bias + I_drive)/tau_m : 1 (unless refractory)
    dge/dt = -ge/tau_e : 1
    dgi/dt = -gi/tau_i : 1
    I_drive : 1
    v_rest : 1
    v_th   : 1
    v_reset: 1
    tau_m  : second
    tau_e  : second
    tau_i  : second
    I_bias : 1
    '''
    G0 = NeuronGroup(Np, eqs, threshold="v>v_th", reset="v=v_reset", refractory=params["t_ref"]*ms, method="euler")
    G1 = NeuronGroup(Np, eqs, threshold="v>v_th", reset="v=v_reset", refractory=params["t_ref"]*ms, method="euler")
    for G in (G0, G1):
        G.v = params["v_reset"]
        G.v_rest = params["v_rest"]
        G.v_th = params["v_th"]
        G.v_reset = params["v_reset"]
        G.tau_m = params["tau_m"]*ms
        G.tau_e = params["tau_e"]*ms
        G.tau_i = params["tau_i"]*ms
        G.I_bias = params["i_bias"]
        G.I_drive = 0.0

    stdp_model = '''
    w : 1
    Apre : 1
    Apost : 1
    taupre : second
    taupost: second
    dApre  : 1
    dApost : 1
    wmax   : 1
    '''
    on_pre = '''
    ge_post += w
    Apre = Apre*exp(-dt/taupre) + dApre
    w = clip(w + Apost, 0, wmax)
    '''
    on_post = '''
    Apost = Apost*exp(-dt/taupost) + dApost
    w = clip(w + Apre, 0, wmax)
    '''
    S0 = Synapses(G_in, G0, model=stdp_model, on_pre=on_pre, on_post=on_post)
    S1 = Synapses(G_in, G1, model=stdp_model, on_pre=on_pre, on_post=on_post)
    S0.connect(True)
    S1.connect(True)
    for S in (S0, S1):
        S.w = np.random.uniform(0.0, params["w_init"], size=len(S.w))
        S.dApre  = params["eta"]
        S.dApost = -params["eta"]*params["beta"]
        S.taupre  = params["tau_pre"]*ms
        S.taupost = params["tau_post"]*ms
        S.wmax = params["wmax"]

    # Inhibition (lateral & cross) via gi_post increments
    I0 = I1 = None; X01 = X10 = None
    if params["inh_w"] > 0:
        I0 = Synapses(G0, G0, model="w_inh:1", on_pre="gi_post += w_inh")
        I0.connect(condition="i!=j"); I0.w_inh = params["inh_w"]
        I1 = Synapses(G1, G1, model="w_inh:1", on_pre="gi_post += w_inh")
        I1.connect(condition="i!=j"); I1.w_inh = params["inh_w"]
    if params["cross_inh_w"] > 0:
        X01 = Synapses(G0, G1, model="w_inh:1", on_pre="gi_post += w_inh")
        X01.connect(True); X01.w_inh = params["cross_inh_w"]
        X10 = Synapses(G1, G0, model="w_inh:1", on_pre="gi_post += w_inh")
        X10.connect(True); X10.w_inh = params["cross_inh_w"]

    M0 = SpikeMonitor(G0)
    M1 = SpikeMonitor(G1)

    net = Network(G_in, G0, G1, S0, S1, M0, M1)
    if I0: net.add(I0, I1)
    if X01: net.add(X01, X10)
    return net, G_in, G0, G1, S0, S1, M0, M1

def normalize_columns(S, target_sum=1.0):
    if target_sum <= 0: return
    w = S.w[:].copy()
    j = S.j[:]
    n_post = int(j.max())+1 if j.size>0 else 0
    if n_post == 0: return
    sums = np.zeros(n_post, dtype=float)
    for idx, jj in enumerate(j): sums[jj] += w[idx]
    scale = np.ones_like(sums)
    nz = sums > 0
    scale[nz] = target_sum / sums[nz]
    for idx, jj in enumerate(j):
        w[idx] = np.clip(w[idx] * scale[jj], 0.0, float(S.wmax[0]))
    S.w[:] = w

def syn_to_matrix(S, C, Np):
    W = np.zeros((C, Np), dtype=float)
    i = np.asarray(S.i[:]); j = np.asarray(S.j[:]); w = np.asarray(S.w[:])
    if i.size: W[i, j] = w
    return W

def cosine_sim(A, B):
    a = A.ravel(); b = B.ravel()
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12: return 0.0
    return float(np.dot(a, b)/(na*nb))

# ---------------------------- Run one epoch ----------------------------
def run_epoch(net, X, y, G_in, G0, G1, S0, S1, M0, M1, params, subset=None, teacher=False):
    C, T = X.shape[1], X.shape[2]
    dt_ms = params["dt_ms"]
    events = []; labels = []
    no_spike = 0; total_spikes = 0

    idx = subset if subset is not None else np.arange(len(X), dtype=int)
    for k in idx:
        x = X[k]; label = int(y[k])
        for G in (G0, G1):
            G.I_drive = 0.0; G.v = params["v_reset"]; G.ge=0; G.gi=0

        # schedule spikes in absolute time (offset by current net.t)
        ch, tbin = np.where(x>0)
        tms = (tbin.astype(float) * dt_ms)
        G_in.set_spikes(ch.astype(int), (tms + float(net.t/ms))*ms)

        if teacher:
            t_first = float(tms.min()) if tms.size>0 else 0.0
            t_on = float(max(0.0, min(T*dt_ms - params["teach_ms"], t_first + params["teach_delay"])))
            if t_on > 0: net.run(t_on*ms)
            if label == 0: G0.I_drive = params["teacher_i"]
            else:          G1.I_drive = params["teacher_i"]
            net.run(params["teach_ms"]*ms)
            G0.I_drive = 0.0; G1.I_drive = 0.0
            rem = max(0.0, T*dt_ms - t_on - params["teach_ms"])
            if rem > 0: net.run(rem*ms)
        else:
            net.run((T*dt_ms)*ms)

        # collect spikes in the last sample window
        t0 = float(net.t/ms) - T*dt_ms
        ids0 = M0.i[(M0.t/ms >= t0) & (M0.t/ms < t0 + T*dt_ms)]
        t0s  = M0.t[(M0.t/ms >= t0) & (M0.t/ms < t0 + T*dt_ms)]/ms
        ids1 = M1.i[(M1.t/ms >= t0) & (M1.t/ms < t0 + T*dt_ms)]
        t1s  = M1.t[(M1.t/ms >= t0) & (M1.t/ms < t0 + T*dt_ms)]/ms

        out_ids = np.concatenate([np.array(ids0), params["Np"] + np.array(ids1)]).astype(int)
        out_t   = np.concatenate([np.array(t0s) - t0, np.array(t1s) - t0]).astype(float)

        events.append((out_ids, out_t)); labels.append(label)
        total_spikes += out_ids.size
        if out_ids.size == 0: no_spike += 1

        normalize_columns(S0, target_sum=params["w_norm"])
        normalize_columns(S1, target_sum=params["w_norm"])

    acc = eval_all(events, labels, params["Np"], tau=params["readout_tau"], N=params["race_N"])
    diag = {"mean_spikes": total_spikes/float(len(idx)+EPS), "no_spike_pct": 100.0*no_spike/float(len(idx))}
    return events, labels, acc, diag

# ---------------------------- Main ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--x-key", default="x")
    ap.add_argument("--label-key", default="label")
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--time", type=int, default=100)
    ap.add_argument("--dt-ms", type=float, default=1.0)

    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--out-pop", type=int, default=8)
    ap.add_argument("--epoch-samples", type=int, default=128)
    ap.add_argument("--balance-train", action="store_true")
    ap.add_argument("--balance-val", action="store_true")

    # neuron
    ap.add_argument("--v-th", type=float, default=0.50)
    ap.add_argument("--v-rest", type=float, default=0.0)
    ap.add_argument("--v-reset", type=float, default=0.0)
    ap.add_argument("--tau-m", type=float, default=20.0)
    ap.add_argument("--tau-e", type=float, default=10.0)
    ap.add_argument("--tau-i", type=float, default=10.0)
    ap.add_argument("--t-ref", type=float, default=2.0)
    ap.add_argument("--i-bias", type=float, default=0.10)

    # STDP
    ap.add_argument("--eta", type=float, default=0.01)
    ap.add_argument("--beta", type=float, default=0.9)
    ap.add_argument("--tau-pre", type=float, default=20.0)
    ap.add_argument("--tau-post", type=float, default=20.0)
    ap.add_argument("--wmax", type=float, default=3.0)
    ap.add_argument("--w-init", type=float, default=0.6)
    ap.add_argument("--w-norm", type=float, default=8.0)

    # inhibition
    ap.add_argument("--inh-w", type=float, default=0.8)
    ap.add_argument("--cross-inh-w", type=float, default=0.6)

    # teacher
    ap.add_argument("--teacher-i", type=float, default=0.35)
    ap.add_argument("--teach-ms", type=float, default=8.0)
    ap.add_argument("--teach-delay", type=float, default=2.0)

    # readout
    ap.add_argument("--readout-tau", type=float, default=30.0)
    ap.add_argument("--race-N", type=int, default=3)

    ap.add_argument("--save", type=str, default="weights_log.npz", help="npz to save weight history & metrics")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    # load + split
    Xall, yall = load_all_npz(Path(args.npz_dir), x_key=args.x_key, label_key=args.label_key)
    idx_tr, idx_va, _ = stratified_split_indices(yall, val_split=args.val_split, test_split=0.0, seed=args.seed)
    Xtr, ytr = Xall[idx_tr], yall[idx_tr]
    Xva, yva = Xall[idx_va], yall[idx_va]
    C, T = Xtr.shape[1], Xtr.shape[2]
    print(f"Loaded train {len(Xtr)}, val {len(Xva)} | shape=({C},{T}) | layout=single-folder")

    params = dict(
        dt_ms=args.dt_ms, v_th=args.v_th, v_rest=args.v_rest, v_reset=args.v_reset,
        tau_m=args.tau_m, tau_e=args.tau_e, tau_i=args.tau_i, t_ref=args.t_ref,
        i_bias=args.i_bias, eta=args.eta, beta=args.beta, tau_pre=args.tau_pre, tau_post=args.tau_post,
        wmax=args.wmax, w_init=args.w_init, w_norm=args.w_norm,
        inh_w=args.inh_w, cross_inh_w=args.cross_inh_w,
        teacher_i=args.teacher_i, teach_ms=args.teach_ms, teach_delay=args.teach_delay,
        readout_tau=args.readout_tau, race_N=args.race_N, Np=args.out_pop
    )

    net, G_in, G0, G1, S0, S1, M0, M1 = build_net(C, args.out_pop, params)
    rng = np.random.default_rng(args.seed)

    # ----- weight history -----
    W0_hist = []
    W1_hist = []
    dmean_hist = []
    cos_prev_hist = []
    cos_first_hist = []
    sat_hi_hist = []
    sat_lo_hist = []

    # initial snapshot
    def syn_to_matrix(S, C, Np):
        W = np.zeros((C, Np), dtype=float)
        i = np.asarray(S.i[:]); j = np.asarray(S.j[:]); w = np.asarray(S.w[:])
        if i.size: W[i, j] = w
        return W

    def cosine_sim(A, B):
        a = A.ravel(); b = B.ravel()
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12: return 0.0
        return float(np.dot(a, b)/(na*nb))

    W0_prev = syn_to_matrix(S0, C, args.out_pop)
    W1_prev = syn_to_matrix(S1, C, args.out_pop)
    W0_hist.append(W0_prev.copy())
    W1_hist.append(W1_prev.copy())
    W0_first = W0_prev.copy()
    W1_first = W1_prev.copy()

    for ep in range(1, args.epochs+1):
        # pick subset
        n = min(args.epoch_samples, len(Xtr))
        if args.balance_train:
            idx0 = np.where(ytr==0)[0]; idx1 = np.where(ytr==1)[0]
            n0 = n//2; n1 = n - n0
            sel = np.concatenate([
                rng.choice(idx0, size=min(n0, len(idx0)), replace=len(idx0)<n0),
                rng.choice(idx1, size=min(n1, len(idx1)), replace=len(idx1)<n1),
            ]); rng.shuffle(sel)
        else:
            sel = rng.choice(np.arange(len(Xtr)), size=n, replace=False)

        # Train & Val
        tr_events, tr_labels, tr_acc, tr_diag = run_epoch(net, Xtr, ytr, G_in, G0, G1, S0, S1, M0, M1, params, subset=sel, teacher=True)
        tr_nt_events, tr_nt_labels, tr_nt_acc, tr_nt_diag = run_epoch(net, Xtr, ytr, G_in, G0, G1, S0, S1, M0, M1, params, subset=sel, teacher=False)
        va_events, va_labels, va_acc, va_diag = run_epoch(net, Xva, yva, G_in, G0, G1, S0, S1, M0, M1, params, subset=np.arange(len(Xva), dtype=int), teacher=False)

        print(f"[Epoch {ep:2d}/{args.epochs}] Train (teach) acc:", {k: round(v,3) for k,v in tr_acc.items()})
        print(f"                     Train (no-teach) acc:", {k: round(v,3) for k,v in tr_nt_acc.items()})
        print(f"                     Val acc:", {k: round(v,3) for k,v in va_acc.items()})
        print(f"           [Diag] train mean spk: {tr_diag['mean_spikes']:.1f} | train no-spike%: {tr_diag['no_spike_pct']:.1f}% | "
              f"val mean spk: {va_diag['mean_spikes']:.1f} | val no-spike%: {va_diag['no_spike_pct']:.1f}%")

        # ----- weight deltas -----
        W0 = syn_to_matrix(S0, C, args.out_pop)
        W1 = syn_to_matrix(S1, C, args.out_pop)
        dmean = 0.5*(np.mean(np.abs(W0 - W0_prev)) + np.mean(np.abs(W1 - W1_prev)))
        cos_prev = 0.5*(cosine_sim(W0, W0_prev) + cosine_sim(W1, W1_prev))
        cos_first = 0.5*(cosine_sim(W0, W0_first) + cosine_sim(W1, W1_first))
        sat_hi = 0.5*(((W0 >= 0.95*args.wmax).mean()) + ((W1 >= 0.95*args.wmax).mean()))
        sat_lo = 0.5*(((W0 <= 1e-4).mean()) + ((W1 <= 1e-4).mean()))

        dmean_hist.append(dmean)
        cos_prev_hist.append(cos_prev)
        cos_first_hist.append(cos_first)
        sat_hi_hist.append(float(sat_hi))
        sat_lo_hist.append(float(sat_lo))
        W0_hist.append(W0.copy()); W1_hist.append(W1.copy())
        W0_prev, W1_prev = W0, W1

        # decay teacher slightly each epoch
        params['teacher_i'] = max(0.1, params['teacher_i']*0.9)

    # save all
    out = Path(args.save)
    np.savez(out,
             W0=np.stack(W0_hist), W1=np.stack(W1_hist),
             dmean=np.array(dmean_hist), cos_prev=np.array(cos_prev_hist),
             cos_first=np.array(cos_first_hist), sat_hi=np.array(sat_hi_hist),
             sat_lo=np.array(sat_lo_hist),
             meta=json.dumps(dict(vars(args))))
    print(f"[Saved] {out} with weight history & metrics.")

if __name__ == "__main__":
    main()
