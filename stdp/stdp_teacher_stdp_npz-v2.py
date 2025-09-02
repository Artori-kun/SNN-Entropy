#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# STDP (teacher-aided) classifier for NPZ spike datasets (one-folder version).
# - Accepts a single folder of .npz files (keys: x_key, label_key)
# - Stratified split into train/val(/test) with --val-split/--test-split
# - Pair-based STDP (no reward); optional short teacher drive aligned to first input spike
# - Reports per-epoch: Train (with teacher), Train (no-teacher), Validation accuracies
#   for multiple readouts (count/weighted/first/raceN/ensemble) + spike diagnostics
#
# Requires: brian2

import argparse, glob
from pathlib import Path
import numpy as np

from brian2 import (ms, defaultclock, start_scope,
                    NeuronGroup, SpikeGeneratorGroup, Synapses,
                    SpikeMonitor, Network)

# ---------------------------- Readouts ----------------------------
EPS = 1e-9

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
    # fallback
    return 0 if (out_ids < Np).sum() >= (out_ids >= Np).sum() else 1

def predict_count(out_ids, Np):
    if out_ids.size == 0: return 0
    return 0 if (out_ids < Np).sum() >= (out_ids >= Np).sum() else 1

def eval_all(events, labels, Np, tau=30.0, N=3):
    y = np.asarray(labels, dtype=int)
    acc = {}
    preds = {}
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

    # Pair-based STDP (rename taupre/taupost to avoid _post name conflict)
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

    # Inhibition
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

# ---------------------------- Run one epoch ----------------------------
def run_epoch(net, X, y, G_in, G0, G1, S0, S1, M0, M1, params, subset=None, teacher=False):
    C, T = X.shape[1], X.shape[2]
    dt_ms = params["dt_ms"]
    events = []; labels = []
    no_spike = 0; total_spikes = 0

    idx = subset if subset is not None else np.arange(len(X), dtype=int)
    for k in idx:
        x = X[k]; label = int(y[k])
        # reset
        for G in (G0, G1):
            G.I_drive = 0.0; G.v = params["v_reset"]; G.ge=0; G.gi=0

        ind, tms = make_spike_indices(x, dt_ms=dt_ms)
        # schedule spikes in absolute time (offset by current net.t)
        G_in.set_spikes(ind, (tms + float(net.t/ms))*ms)

        if teacher:
            # align teacher window to first input spike (+ delay)
            t_first = float(tms.min()) if tms.size>0 else 0.0
            t_on = float(max(0.0, min(T*dt_ms - params["teach_ms"], t_first + params["teach_delay"])))
            # pre-teach segment
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

        # normalize weights per column
        normalize_columns(S0, target_sum=params["w_norm"])
        normalize_columns(S1, target_sum=params["w_norm"])

    acc = eval_all(events, labels, params["Np"], tau=params["readout_tau"], N=params["race_N"])
    diag = {"mean_spikes": total_spikes/float(len(idx)+EPS), "no_spike_pct": 100.0*no_spike/float(len(idx))}
    return events, labels, acc, diag

# ---------------------------- Main ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", required=True, help="Folder with all .npz files (single folder).")
    ap.add_argument("--x-key", default="x")
    ap.add_argument("--label-key", default="label")
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--test-split", type=float, default=0.0)  # unused for now
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
    ap.add_argument("--v-th", type=float, default=0.48)
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
    ap.add_argument("--w-norm", type=float, default=8.0, help="Column L1 norm target; set 0 to disable normalize")

    # inhibition
    ap.add_argument("--inh-w", type=float, default=0.8)
    ap.add_argument("--cross-inh-w", type=float, default=0.6)

    # teacher
    ap.add_argument("--teacher-i", type=float, default=0.4)
    ap.add_argument("--teach-ms", type=float, default=8.0)
    ap.add_argument("--teach-delay", type=float, default=2.0)

    # readout
    ap.add_argument("--readout-tau", type=float, default=30.0)
    ap.add_argument("--race-N", type=int, default=3)

    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    # load + split
    Xall, yall = load_all_npz(Path(args.npz_dir), x_key=args.x_key, label_key=args.label_key)
    idx_tr, idx_va, _ = stratified_split_indices(yall, val_split=args.val_split, test_split=args.test_split, seed=args.seed)
    Xtr, ytr = Xall[idx_tr], yall[idx_tr]
    Xva, yva = Xall[idx_va], yall[idx_va]
    C, T = Xtr.shape[1], Xtr.shape[2]
    print(f"Loaded train {len(Xtr)}, val {len(Xva)} | shape=({C},{T}) | layout=single-folder")

    # params
    params = dict(
        dt_ms=args.dt_ms, v_th=args.v_th, v_rest=args.v_rest, v_reset=args.v_reset,
        tau_m=args.tau_m, tau_e=args.tau_e, tau_i=args.tau_i, t_ref=args.t_ref,
        i_bias=args.i_bias, eta=args.eta, beta=args.beta, tau_pre=args.tau_pre, tau_post=args.tau_post,
        wmax=args.wmax, w_init=args.w_init, w_norm=args.w_norm,
        inh_w=args.inh_w, cross_inh_w=args.cross_inh_w,
        teacher_i=args.teacher_i, teach_ms=args.teach_ms, teach_delay=args.teach_delay,
        readout_tau=args.readout_tau, race_N=args.race_N, Np=args.out_pop
    )

    # build net
    net, G_in, G0, G1, S0, S1, M0, M1 = build_net(C, args.out_pop, params)

    rng = np.random.default_rng(args.seed)

    for ep in range(1, args.epochs+1):
        # balanced subset for training
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

        # TRAIN (with teacher) and TRAIN (no-teacher)
        tr_events, tr_labels, tr_acc, tr_diag = run_epoch(net, Xtr, ytr, G_in, G0, G1, S0, S1, M0, M1, params, subset=sel, teacher=True)
        tr_nt_events, tr_nt_labels, tr_nt_acc, tr_nt_diag = run_epoch(net, Xtr, ytr, G_in, G0, G1, S0, S1, M0, M1, params, subset=sel, teacher=False)

        # VAL
        if args.balance_val:
            idv0 = np.where(yva==0)[0]; idv1 = np.where(yva==1)[0]
            nV = len(Xva); nv0 = nV//2; nv1 = nV - nv0
            selV = np.concatenate([
                rng.choice(idv0, size=min(nv0, len(idv0)), replace=len(idv0)<nv0),
                rng.choice(idv1, size=min(nv1, len(idv1)), replace=len(idv1)<nv1),
            ]); rng.shuffle(selV)
        else:
            selV = np.arange(len(Xva), dtype=int)
        va_events, va_labels, va_acc, va_diag = run_epoch(net, Xva, yva, G_in, G0, G1, S0, S1, M0, M1, params, subset=selV, teacher=False)

        # Imbalance metric on val (0≈tie, 1≈one-sided)
        def _imb(evts):
            vals=[]
            for ids,t in evts:
                if len(ids)==0: continue
                n0=(ids<params['Np']).sum(); n1=(ids>=params['Np']).sum(); s=n0+n1
                if s>0: vals.append(abs(n0-n1)/s)
            return 0.0 if not vals else float(sum(vals)/len(vals))
        va_imb = _imb(va_events)

        # Population means for diagnostics
        def _pop_means(evts, Np):
            if not evts: return 0.0, 0.0
            s0=s1=0.0
            for ids,t in evts:
                ids = np.asarray(ids)
                s0 += (ids < Np).sum()
                s1 += (ids >= Np).sum()
            n = max(len(evts), 1)
            return s0/float(n), s1/float(n)
        va_m0, va_m1 = _pop_means(va_events, params['Np'])

        print(f"[Epoch {ep:2d}/{args.epochs}] Train acc (teach) | "
              f"count: {tr_acc['count']:.3f} | weighted: {tr_acc['weighted']:.3f} | first: {tr_acc['first']:.3f} | raceN: {tr_acc['raceN']:.3f} | ensemble: {tr_acc['ensemble']:.3f}")
        print(f"                     Train acc (no-teach) | "
              f"count: {tr_nt_acc['count']:.3f} | weighted: {tr_nt_acc['weighted']:.3f} | first: {tr_nt_acc['first']:.3f} | raceN: {tr_nt_acc['raceN']:.3f} | ensemble: {tr_nt_acc['ensemble']:.3f}")
        print(f"                     [Val] acc | count: {va_acc['count']:.3f} | weighted: {va_acc['weighted']:.3f} | first: {va_acc['first']:.3f} | raceN: {va_acc['raceN']:.3f} | ensemble: {va_acc['ensemble']:.3f} | imb: {va_imb:.3f} | m0: {va_m0:.1f} | m1: {va_m1:.1f}")
        print(f"           [Diag] train mean spk: {tr_diag['mean_spikes']:.1f} | train no-spike%: {tr_diag['no_spike_pct']:.1f}% | "
              f"val mean spk: {va_diag['mean_spikes']:.1f} | val no-spike%: {va_diag['no_spike_pct']:.1f}%")

        # Teacher decay (to avoid overfitting to teacher)
        params['teacher_i'] = max(0.1, params['teacher_i']*0.9)

        # Simple LR decay for STDP step
        for S in (S0, S1):
            S.dApre[:]  *= 0.98
            S.dApost[:] *= 0.98

        # --- Per-pop homeostasis: if one population dominates, nudge thresholds ---
        if va_imb > 0.6 and (va_m0 + va_m1) > 0:
            if va_m0 > va_m1:
                new_v0 = max(0.0, float(G0.v_th[0]) + 0.02)
                new_v1 = max(0.0, float(G1.v_th[0]) - 0.02)
                G0.v_th = new_v0; G1.v_th = new_v1
            else:
                new_v0 = max(0.0, float(G0.v_th[0]) - 0.02)
                new_v1 = max(0.0, float(G1.v_th[0]) + 0.02)
                G0.v_th = new_v0; G1.v_th = new_v1

        # Auto boost/trim if totally silent or saturated (global)
        if va_diag['no_spike_pct'] > 50.0:
            params['v_th'] = max(0.30, params['v_th']-0.02)
            G0.v_th = params['v_th']; G1.v_th = params['v_th']
        elif va_diag['mean_spikes'] > 80.0:
            params['v_th'] = min(1.0, params['v_th']+0.02)
            G0.v_th = params['v_th']; G1.v_th = params['v_th']

    print("[Done]")

if __name__ == "__main__":
    main()
