
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Brian2 STDP classifier (unsupervised) with output population + train/val/test + validation spike logging.

Data: folder of .npz (one sample each):
  - x: (C, T) binary spike matrix
  - label: scalar 0/1

Output population: K neurons per class (total 2K). After each epoch, build a label mapping per neuron
from the training set responses (no learning), then evaluate validation accuracy and (optionally) log spikes.
At the end, rebuild mapping on the full training set and evaluate test accuracy.

Learning: pair-based STDP (no reward, no teacher)
  - On presyn spike:   w += eta * Aplus  * xpost     (LTP if post recently spiked)
  - On postsyn spike:  w -= eta * Aminus * xpre      (LTD if pre recently spiked)
with pre/post traces xpre/xpost following exponential decays (taus taupre/taupost).

Requires: brian2, numpy
"""

import argparse, json
from pathlib import Path
import numpy as np
from brian2 import (ms, prefs, defaultclock,
                    SpikeGeneratorGroup, NeuronGroup, Synapses, Network,
                    SpikeMonitor, clip)

def parse_args():
    p = argparse.ArgumentParser(description="Brian2 STDP population classifier with validation")
    p.add_argument("--npz-dir", type=str, required=True)
    p.add_argument("--x-key", type=str, default="x")
    p.add_argument("--label-key", type=str, default="label")
    p.add_argument("--channels", type=int, default=None)
    p.add_argument("--time", type=int, default=None)
    p.add_argument("--strict-shape", action="store_true")
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gap", type=float, default=25.0)
    p.add_argument("--out_k", type=int, default=8)
    p.add_argument("--v_th", type=float, default=0.6)
    p.add_argument("--tau_mem", type=float, default=20.0)
    p.add_argument("--tau_syn", type=float, default=10.0)
    p.add_argument("--inh_tau", type=float, default=10.0)
    p.add_argument("--refractory", type=float, default=5.0)
    p.add_argument("--inh_w", type=float, default=1.5)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--Aplus", type=float, default=1.0)
    p.add_argument("--Aminus", type=float, default=1.0)
    p.add_argument("--tau_pre", type=float, default=60.0)
    p.add_argument("--tau_post", type=float, default=60.0)
    p.add_argument("--wmax", type=float, default=2.0)
    p.add_argument("--norm_per_post", action="store_true")
    p.add_argument("--train_split", type=float, default=0.8)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--val_log", action="store_true")
    p.add_argument("--report", type=str, default="./stdp_report")
    return p.parse_args()

def load_npz_folder(folder, x_key="x", label_key="label", C=None, T=None, strict=False):
    folder = Path(folder); files = sorted(folder.glob("*.npz"))
    if not files: raise FileNotFoundError(f"No .npz in {folder}")
    X, y = [], []
    for f in files:
        data = np.load(f, allow_pickle=False)
        if x_key not in data or label_key not in data:
            print(f"[WARN] Skip {f.name}: missing keys"); continue
        x = data[x_key]; lbl = data[label_key]
        lbl = int(lbl) if not isinstance(lbl, np.ndarray) else int(np.ravel(lbl)[0])
        if x.ndim != 2: 
            print(f"[WARN] Skip {f.name}: x must be 2D (C,T), got {x.shape}"); continue
        Cx, Tx = x.shape
        if strict and ((C is not None and Cx != C) or (T is not None and Tx != T)):
            print(f"[WARN] Skip {f.name}: expected ({C},{T}), got {x.shape}"); continue
        X.append(x.astype(np.int8)); y.append(lbl)
    if not X: raise RuntimeError("No valid samples loaded.")
    X, y = np.array(X, dtype=np.int8), np.array(y, dtype=int)
    if C is None or T is None: C, T = X.shape[1], X.shape[2]
    return X, y, int(C), int(T)

def spikes_from_matrix(binmat, dt_ms):
    ch, t = np.where(binmat > 0)
    if ch.size == 0: return np.array([], dtype=int), np.array([], dtype=float)
    times_ms = (t.astype(np.float64) * dt_ms)
    return ch.astype(int), times_ms

def build_network(C, dt_ms, params):
    K = params['out_k']; Nout = 2 * K
    defaultclock.dt = dt_ms * ms
    G_in = SpikeGeneratorGroup(C, indices=np.array([], dtype=int), times=np.array([])*ms, name='G_in')
    eqs = '''
    dv/dt = (-v + ge - gi) / tau_m : 1 (unless refractory)
    dge/dt = -ge / tau_e : 1
    dgi/dt = -gi / tau_i : 1
    tau_m : second
    tau_e : second
    tau_i : second
    v_th : 1
    '''
    G_out = NeuronGroup(Nout, eqs, threshold='v>v_th', reset='v=0',
                        refractory=params['refractory']*ms, method='euler', name='outpop')
    G_out.tau_m = params['tau_mem'] * ms
    G_out.tau_e = params['tau_syn'] * ms
    G_out.tau_i = params['inh_tau'] * ms
    G_out.v_th = params['v_th']; G_out.v = 0

    S = Synapses(G_in, G_out,
                 model='''
                 w : 1
                 dxpre/dt = -xpre/taupre : 1 (clock-driven)
                 dxpost/dt = -xpost/taupost : 1 (clock-driven)
                 taupre : second
                 taupost : second
                 Aplus : 1
                 Aminus : 1
                 eta : 1
                 wmax : 1
                 ''',
                 on_pre='''
                 ge_post += w
                 xpre += Aplus
                 w = clip(w + eta * xpost, 0, wmax)
                 ''',
                 on_post='''
                 xpost += Aminus
                 w = clip(w - eta * xpre, 0, wmax)
                 ''',
                 name='S_in_out')
    S.connect(True)
    rng = np.random.default_rng(params['seed'])
    S.w = rng.uniform(0.0, 0.3, size=len(S))
    S.taupre = params['tau_pre'] * ms
    S.taupost = params['tau_post'] * ms
    S.Aplus = params['Aplus']; S.Aminus = params['Aminus']
    S.eta = params['lr']; S.wmax = params['wmax']

    S_inh = Synapses(G_out, G_out, on_pre='gi_post += w_inh', model='w_inh:1', name='S_lateral_inh')
    S_inh.connect(condition='i!=j'); S_inh.w_inh = params['inh_w']

    M_out = SpikeMonitor(G_out, name='M_out')
    net = Network(G_in, G_out, S, S_inh, M_out)
    return net, G_in, G_out, S, M_out, K

def class_counts_by_mapping(M_out, t_start, t_end, mapping, Nout):
    mask = (M_out.t/ms >= t_start) & (M_out.t/ms < t_end)
    ids = np.array(M_out.i[mask], dtype=int)
    if ids.size == 0: return [0, 0]
    cls_ids = mapping[ids]
    c0 = int(np.sum(cls_ids == 0)); c1 = int(np.sum(cls_ids == 1))
    return [c0, c1]

def per_neuron_hist(M_out, t_start, t_end, Nout):
    mask = (M_out.t/ms >= t_start) & (M_out.t/ms < t_end)
    ids = np.array(M_out.i[mask], dtype=int)
    if ids.size == 0: return np.zeros(Nout, dtype=int)
    return np.bincount(ids, minlength=Nout).astype(int)

def train_and_eval(X, y, C, T, args):
    dt_ms = float(args.dt); Tdur_ms = T * dt_ms; gap_ms = float(args.gap)
    n = len(X); n_train = int(np.floor(args.train_split * n)); n_val = int(np.floor(args.val_split * n))
    rng = np.random.default_rng(args.seed); idx = np.arange(n); rng.shuffle(idx)
    train_idx = idx[:n_train]; val_idx = idx[n_train:n_train+n_val]; test_idx = idx[n_train+n_val:]

    params = dict(seed=args.seed, tau_mem=args.tau_mem, tau_syn=args.tau_syn, inh_tau=args.inh_tau,
                  refractory=args.refractory, tau_pre=args.tau_pre, tau_post=args.tau_post,
                  Aplus=args.Aplus, Aminus=args.Aminus, lr=args.lr, wmax=args.wmax,
                  v_th=args.v_th, inh_w=args.inh_w, out_k=args.out_k)
    net, G_in, G_out, S, M_out, K = build_network(C, dt_ms, params)
    Nout = 2 * K

    def normalize_per_post():
        if not args.norm_per_post: return
        for j in range(Nout):
            idx_j = np.where(S.j[:] == j)[0]
            if idx_j.size > 0:
                s = float(np.sum(S.w[idx_j]))
                if s > 0: S.w[idx_j] = S.w[idx_j] / s

    def run_trial(binmat, learn=True):
        S.eta = args.lr if learn else 0.0
        t_now = float(defaultclock.t/ms)
        inds, times_ms = spikes_from_matrix(binmat, dt_ms)
        G_in.set_spikes(inds, (t_now + times_ms) * ms)
        G_out.v = 0; G_out.ge = 0; G_out.gi = 0
        t_start = t_now; t_end = t_now + Tdur_ms
        net.run((Tdur_ms + gap_ms) * ms, report=None)
        return t_start, t_end

    def build_label_mapping(indices):
        counts = np.zeros((Nout, 2), dtype=int)
        for i in indices:
            binmat, label = X[i], int(y[i])
            t_start, t_end = run_trial(binmat, learn=False)
            hist = per_neuron_hist(M_out, t_start, t_end, Nout)
            counts[:, label] += hist
        mapping = np.where(counts[:,1] > counts[:,0], 1, 0).astype(int)
        return mapping

    for ep in range(args.epochs):
        rng.shuffle(train_idx)
        for i in train_idx:
            binmat = X[i]
            run_trial(binmat, learn=True)
            normalize_per_post()

        mapping = build_label_mapping(train_idx)

        val_correct = 0; val_total = 0
        val_dir = Path(args.report) / "val_logs" / f"epoch_{ep+1:03d}"
        if args.val_log: val_dir.mkdir(parents=True, exist_ok=True)
        for i_val in val_idx:
            binmat_v, label_v = X[i_val], int(y[i_val])
            t_start, t_end = run_trial(binmat_v, learn=False)
            counts_v = class_counts_by_mapping(M_out, t_start, t_end, mapping, Nout)
            y_pred_v = int(np.argmax(counts_v)); val_correct += int(y_pred_v == label_v); val_total += 1
            if args.val_log:
                mask = (M_out.t/ms >= t_start) & (M_out.t/ms < t_end)
                out_neuron = np.array(M_out.i[mask], dtype=int)
                out_t_ms = np.array((M_out.t[mask]/ms) - t_start, dtype=float)
                np.savez_compressed(val_dir / f"sample_{int(i_val):04d}.npz",
                                    out_neuron=out_neuron, out_t_ms=out_t_ms,
                                    label=label_v, counts=np.array(counts_v, dtype=int))
        val_acc = val_correct / max(1, val_total)
        print(f"[Epoch {ep+1}/{args.epochs}] Val acc: {val_acc:.3f} on {val_total} samples")

    final_mapping = build_label_mapping(train_idx)

    test_correct = 0; test_total = 0
    for i in test_idx:
        binmat_t, label_t = X[i], int(y[i])
        t_start, t_end = run_trial(binmat_t, learn=False)
        counts_t = class_counts_by_mapping(M_out, t_start, t_end, final_mapping, Nout)
        y_pred_t = int(np.argmax(counts_t)); test_correct += int(y_pred_t == label_t); test_total += 1
    test_acc = test_correct / max(1, test_total)
    print(f"[TEST] Accuracy: {test_acc:.3f} on {test_total} samples")

    return {"val_acc_last": float(val_acc), "test_acc": float(test_acc),
            "n_train": int(len(train_idx)), "n_val": int(len(val_idx)), "n_test": int(len(test_idx))}, S, K

def main():
    args = parse_args()
    out_dir = Path(args.report); out_dir.mkdir(parents=True, exist_ok=True)
    X, y, C, T = load_npz_folder(args.npz_dir, args.x_key, args.label_key,
                                 C=args.channels, T=args.time, strict=args.strict_shape)
    print(f"Loaded {len(X)} samples, shape = ({C},{T}), labels 0/1")
    prefs.codegen.target = 'numpy'
    metrics, S, K = train_and_eval(X, y, C, T, args)
    np.save(out_dir / "weights_final.npy", S.w[:])
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f: json.dump(metrics, f, indent=2)
    groups = {"out_k": int(K), "group_class0": [int(i) for i in range(0, K)],
              "group_class1": [int(i) for i in range(K, 2*K)]}
    with open(out_dir / "outpop_groups.json", "w", encoding="utf-8") as f: json.dump(groups, f, indent=2)
    print("Saved:", (out_dir / "weights_final.npy").resolve())

if __name__ == "__main__":
    main()
