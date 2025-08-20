
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STDP (unsupervised) HIDDEN layer + population OUTPUT classifier
- Input -> Hidden (G_in -> G_h): pair-based STDP with lateral inhibition (unsupervised)
- Hidden -> Output (G_h -> G_out): weights rebuilt from hidden neurons' class preference each epoch
- Population output (2 classes, each out_pop neurons) with lateral inhibition
- Train/Val/Test splits (+ options to force balanced Val/Test)
- Optional balanced subset per epoch
- Readouts: count / time-weighted / first / race-to-N
"""
import argparse, json
from pathlib import Path
import numpy as np
from brian2 import (ms, prefs, defaultclock,
                    SpikeGeneratorGroup, NeuronGroup, Synapses, Network,
                    SpikeMonitor, clip)

# -------------------------
# CLI
# -------------------------

def parse_args():
    p = argparse.ArgumentParser(description="STDP hidden + population output classifier (unsupervised hidden)")
    # Data
    p.add_argument("--npz-dir", type=str, required=True)
    p.add_argument("--x-key", type=str, default="x")
    p.add_argument("--label-key", type=str, default="label")
    p.add_argument("--channels", type=int, default=None)
    p.add_argument("--time", type=int, default=None)
    p.add_argument("--strict-shape", action="store_true")

    # Timing
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--pretrain_epochs", type=int, default=10, help="Epochs with STDP active (after that, freeze hidden)")
    p.add_argument("--seed", type=int, default=42)

    # Hidden size & neuron params
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--v_th_h", type=float, default=0.6)
    p.add_argument("--tau_mem_h", type=float, default=20.0)
    p.add_argument("--tau_syn_h", type=float, default=10.0)
    p.add_argument("--inh_w_h", type=float, default=2.2)
    p.add_argument("--inh_tau_h", type=float, default=10.0)
    p.add_argument("--refractory_h", type=float, default=6.0)
    p.add_argument("--i_bias_h", type=float, default=0.02)

    # STDP (Input->Hidden)
    p.add_argument("--eta_h", type=float, default=0.005)
    p.add_argument("--Apre_h", type=float, default=1.0)
    p.add_argument("--Apost_h", type=float, default=1.0)
    p.add_argument("--tau_pre_h", type=float, default=40.0)
    p.add_argument("--tau_post_h", type=float, default=40.0)
    p.add_argument("--wmax_h", type=float, default=1.5)
    p.add_argument("--norm_h", action="store_true", help="L1 normalise incoming weights per hidden neuron after each sample")

    # Output population
    p.add_argument("--out_pop", type=int, default=8)
    p.add_argument("--v_th_o", type=float, default=0.6)
    p.add_argument("--tau_mem_o", type=float, default=20.0)
    p.add_argument("--tau_syn_o", type=float, default=10.0)
    p.add_argument("--inh_w_o", type=float, default=2.0)
    p.add_argument("--inh_tau_o", type=float, default=10.0)
    p.add_argument("--refractory_o", type=float, default=5.0)
    p.add_argument("--i_bias_o", type=float, default=0.0)

    # Hidden->Out weight mapping
    p.add_argument("--w_out_base", type=float, default=0.05)
    p.add_argument("--w_out_gain", type=float, default=0.7)
    p.add_argument("--jitter_out", type=float, default=0.02, help="random jitter added to h->out weights")

    # Scheduling
    p.add_argument("--train_split", type=float, default=0.8)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--balance_val", action="store_true")
    p.add_argument("--balance_test", action="store_true")

    # Subset per epoch (balanced)
    p.add_argument("--epoch_samples", type=int, default=0, help="If >0, #train samples per epoch (balanced per class)")

    # Readout options
    p.add_argument("--readout_tau", type=float, default=30.0, help="Tau (ms) for time-weighted count")
    p.add_argument("--race_N", type=int, default=3, help="Race-to-N threshold")

    p.add_argument("--report", type=str, default="./stdp_hidden_report")
    return p.parse_args()

# -------------------------
# Data
# -------------------------

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

# -------------------------
# Split helpers
# -------------------------

def make_balanced_splits(y, train_split, val_split, balance_val, balance_test, rng):
    n = len(y); idx0 = np.where(y==0)[0]; idx1 = np.where(y==1)[0]
    rng.shuffle(idx0); rng.shuffle(idx1)
    n_val_total = int(np.floor(val_split * n))
    if balance_val:
        n_val_per = min(len(idx0), len(idx1), n_val_total//2)
        val_idx = np.concatenate([idx0[:n_val_per], idx1[:n_val_per]])
        idx0 = idx0[n_val_per:]; idx1 = idx1[n_val_per:]
    else:
        n_val0 = int(round(n_val_total * len(idx0)/(len(idx0)+len(idx1)))); n_val1 = n_val_total - n_val0
        val_idx = np.concatenate([idx0[:n_val0], idx1[:n_val1]])
        idx0 = idx0[n_val0:]; idx1 = idx1[n_val1:]
    remaining = np.concatenate([idx0, idx1]); rng.shuffle(remaining)
    n_train_total = int(np.floor(train_split * n)); n_train_total = min(n_train_total, len(remaining))
    train_idx = remaining[:n_train_total]; rest = remaining[n_train_total:]
    if balance_test and len(rest)>0:
        r0 = rest[y[rest]==0]; r1 = rest[y[rest]==1]; rng.shuffle(r0); rng.shuffle(r1)
        n_test_per = min(len(r0), len(r1))
        test_idx = np.concatenate([r0[:n_test_per], r1[:n_test_per]])
        leftovers = np.concatenate([r0[n_test_per:], r1[n_test_per:]])
        train_idx = np.concatenate([train_idx, leftovers])
    else:
        test_idx = rest
    rng.shuffle(train_idx); rng.shuffle(val_idx); rng.shuffle(test_idx)
    return train_idx.astype(int), val_idx.astype(int), test_idx.astype(int)

# -------------------------
# Readouts
# -------------------------

def predict_count(counts):
    return int(np.argmax(counts))

def predict_weighted(out_ids, out_t_ms, Np, tau_ms):
    if out_ids.size == 0: return 0
    w0 = np.exp(-out_t_ms[out_ids <  Np] / float(tau_ms)).sum()
    w1 = np.exp(-out_t_ms[out_ids >= Np] / float(tau_ms)).sum()
    return 0 if w0 >= w1 else 1

def predict_first_spike(out_ids, out_t_ms, Np):
    if out_ids.size == 0: return 0
    has0 = np.any(out_ids < Np); has1 = np.any(out_ids >= Np)
    t0 = out_t_ms[out_ids < Np].min() if has0 else np.inf
    t1 = out_t_ms[out_ids >= Np].min() if has1 else np.inf
    if np.isinf(t0) and np.isinf(t1): return 0
    return 0 if t0 <= t1 else 1

def predict_race_to_N(out_ids, out_t_ms, Np, N=3):
    if out_ids.size == 0: return 0
    order = np.argsort(out_t_ms)
    c0=c1=0
    for k in order:
        if out_ids[k] < Np: c0 += 1
        else: c1 += 1
        if c0>=N or c1>=N:
            return 0 if c0>c1 else 1
    return 0 if (out_ids < Np).sum() >= (out_ids >= Np).sum() else 1

# -------------------------
# Network
# -------------------------

def build_network(C, T, dt_ms, args, rng):
    Np = args.out_pop; Nout = 2*Np; Nh = args.hidden
    defaultclock.dt = dt_ms * ms

    # Input
    G_in = SpikeGeneratorGroup(C, indices=np.array([], dtype=int), times=np.array([])*ms, name='G_in')

    # Hidden LIF
    eqs_h = '''
    dv/dt = (-v + ge - gi + Ibias) / tau_m : 1 (unless refractory)
    dge/dt = -ge / tau_e : 1
    dgi/dt = -gi / tau_i : 1
    Ibias : 1
    tau_m : second
    tau_e : second
    tau_i : second
    v_th : 1
    '''
    G_h = NeuronGroup(Nh, eqs_h,
                      threshold='v>v_th',
                      reset='v=0',
                      refractory=args.refractory_h*ms, method='euler', name='G_h')
    G_h.tau_m = args.tau_mem_h * ms; G_h.tau_e = args.tau_syn_h * ms; G_h.tau_i = args.inh_tau_h * ms
    G_h.v_th = args.v_th_h; G_h.v = 0; G_h.ge=0; G_h.gi=0; G_h.Ibias = args.i_bias_h

    # STDP In->Hidden
    S_in_h = Synapses(G_in, G_h,
        model='''
        w : 1
        wmax : 1
        dpretr/dt = -pretr / taupre : 1 (clock-driven)
        dposttr/dt = -posttr / taupost : 1 (clock-driven)
        taupre : second
        taupost : second
        Apre : 1
        Apost : 1
        eta : 1
        ''',
        on_pre='''
        ge_post += w
        pretr += Apre
        w = clip(w - eta * posttr, 0, wmax)
        ''',
        on_post='''
        posttr += Apost
        w = clip(w + eta * pretr, 0, wmax)
        ''',
        name='S_in_h')
    S_in_h.connect(True)
    S_in_h.w = rng.uniform(0.0, 0.3, size=len(S_in_h))
    S_in_h.wmax = args.wmax_h
    S_in_h.taupre = args.tau_pre_h * ms
    S_in_h.taupost = args.tau_post_h * ms
    S_in_h.Apre = args.Apre_h
    S_in_h.Apost = args.Apost_h
    S_in_h.eta = args.eta_h

    # Lateral inhibition in hidden
    S_inh_h = Synapses(G_h, G_h, on_pre='gi_post += w_inh', model='w_inh:1', name='S_inh_h')
    S_inh_h.connect(condition='i!=j'); S_inh_h.w_inh = args.inh_w_h

    # Output LIF
    eqs_o = '''
    dv/dt = (-v + ge - gi + Ibias) / tau_m : 1 (unless refractory)
    dge/dt = -ge / tau_e : 1
    dgi/dt = -gi / tau_i : 1
    Ibias : 1
    tau_m : second
    tau_e : second
    tau_i : second
    v_th : 1
    '''
    G_out = NeuronGroup(Nout, eqs_o,
                        threshold='v>v_th',
                        reset='v=0',
                        refractory=args.refractory_o*ms, method='euler', name='G_out')
    G_out.tau_m = args.tau_mem_o * ms; G_out.tau_e = args.tau_syn_o * ms; G_out.tau_i = args.inh_tau_o * ms
    G_out.v_th = args.v_th_o; G_out.v = 0; G_out.ge=0; G_out.gi=0; G_out.Ibias = args.i_bias_o

    # Hidden -> Output (static weights set from preferences each epoch)
    S_h_out = Synapses(G_h, G_out, model='w:1', on_pre='ge_post += w', name='S_h_out')
    S_h_out.connect(True)
    # init small random
    S_h_out.w = args.w_out_base + args.jitter_out * (rng.random(len(S_h_out)) - 0.5)

    # Output lateral inhibition
    S_inh_o = Synapses(G_out, G_out, on_pre='gi_post += w_inh', model='w_inh:1', name='S_inh_o')
    S_inh_o.connect(condition='i!=j'); S_inh_o.w_inh = args.inh_w_o

    # Monitors
    M_h = SpikeMonitor(G_h, name='M_h')
    M_o = SpikeMonitor(G_out, name='M_out')

    net = Network(G_in, G_h, G_out, S_in_h, S_h_out, S_inh_h, S_inh_o, M_h, M_o)
    return net, G_in, G_h, G_out, S_in_h, S_h_out, M_h, M_o, Np

# -------------------------
# Training / Evaluation
# -------------------------

def train_and_eval(X, y, C, T, args):
    rng = np.random.default_rng(args.seed)
    dt_ms = float(args.dt); defaultclock.dt = dt_ms * ms
    n = len(X)
    train_idx, val_idx, test_idx = make_balanced_splits(y, args.train_split, args.val_split, args.balance_val, args.balance_test, rng)

    # Split summary
    def _sum(y_idx):
        if len(y_idx)==0: return (0,0,0.0)
        c0 = int(np.sum(y[y_idx]==0)); c1 = int(np.sum(y[y_idx]==1)); ratio = c1 / max(1, c0+c1)
        return c0, c1, ratio
    tr0,tr1,trr = _sum(train_idx); va0,va1,var = _sum(val_idx); te0,te1,ter = _sum(test_idx)
    print(f"[Split] Train: {len(train_idx)} (y0={tr0}, y1={tr1}, frac1={trr:.2f}) | Val: {len(val_idx)} (y0={va0}, y1={va1}, frac1={var:.2f}) | Test: {len(test_idx)} (y0={te0}, y1={te1}, frac1={ter:.2f})")

    net, G_in, G_h, G_out, S_in_h, S_h_out, M_h, M_o, Np = build_network(C, T, dt_ms, args, rng)
    Nout = 2*Np; Tdur_ms = T * dt_ms

    def set_input_spikes(binmat):
        t_now = float(defaultclock.t/ms)
        inds, times_ms = spikes_from_matrix(binmat, dt_ms)
        if inds.size > 0:
            G_in.set_spikes(inds, (t_now + times_ms)*ms)
        return t_now

    def counts_by_class(t0, t1):
        mask = (M_o.t/ms >= t0) & (M_o.t/ms < t1)
        ids = np.array(M_o.i[mask], dtype=int)
        if ids.size == 0: return [0,0]
        c0 = int(np.sum((ids>=0) & (ids < Np))); c1 = int(np.sum((ids>=Np) & (ids < 2*Np)))
        return [c0, c1]

    def update_out_weights_from_hidden_pref(hid_c0, hid_c1):
        eps = 1e-6
        pref = (hid_c1 - hid_c0) / (hid_c1 + hid_c0 + eps)  # [-1,1] per hidden neuron
        # same weight from hidden j to all neurons in a class-pop (with jitter)
        w = S_h_out.w[:]
        j_arr = S_h_out.j[:]; i_arr = S_h_out.i[:]
        w0_per_h = args.w_out_base + args.w_out_gain * np.clip(-pref, 0, 1)
        w1_per_h = args.w_out_base + args.w_out_gain * np.clip(+pref, 0, 1)
        for idx_syn in range(len(S_h_out)):
            j = j_arr[idx_syn]; i = i_arr[idx_syn]
            base = w0_per_h[j] if i < Np else w1_per_h[j]
            w[idx_syn] = base + args.jitter_out * (rng.random()-0.5)
        S_h_out.w = clip(w, 0, args.w_out_base + args.w_out_gain + 2*args.jitter_out)

    # Training loop
    for ep in range(args.epochs):
        # Subset
        if args.epoch_samples and args.epoch_samples > 0:
            idx0 = train_idx[y[train_idx]==0]; idx1 = train_idx[y[train_idx]==1]
            m = min(args.epoch_samples//2, len(idx0), len(idx1))
            ep_idx = np.concatenate([rng.choice(idx0, m, replace=False), rng.choice(idx1, m, replace=False)])
            rng.shuffle(ep_idx)
        else:
            ep_idx = train_idx.copy(); rng.shuffle(ep_idx)

        hid_c0 = np.zeros(args.hidden, dtype=float)
        hid_c1 = np.zeros(args.hidden, dtype=float)
        train_correct_c = train_correct_w = train_correct_f = train_correct_r = 0
        train_total = 0

        # Train subset (STDP on in->hidden during pretrain epochs)
        for ii in ep_idx:
            binmat = X[ii]; label = int(y[ii])
            t0 = set_input_spikes(binmat)
            G_h.v = 0; G_h.ge = 0; G_h.gi = 0
            G_out.v = 0; G_out.ge = 0; G_out.gi = 0
            S_in_h.eta = args.eta_h if ep < args.pretrain_epochs else 0.0
            net.run(Tdur_ms * ms, report=None)

            # Hidden spikes count per class
            maskh = (M_h.t/ms >= t0) & (M_h.t/ms < t0 + Tdur_ms)
            if np.any(maskh):
                ids_h = np.array(M_h.i[maskh], dtype=int)
                if label==0: np.add.at(hid_c0, ids_h, 1)
                else:        np.add.at(hid_c1, ids_h, 1)

            # Output prediction
            masko = (M_o.t/ms >= t0) & (M_o.t/ms < t0 + Tdur_ms)
            ids_o = np.array(M_o.i[masko], dtype=int)
            t_o = np.array((M_o.t[masko]/ms) - t0, dtype=float)
            counts = counts_by_class(t0, t0+Tdur_ms)
            y_c = predict_count(counts)
            y_w = predict_weighted(ids_o, t_o, Np, args.readout_tau)
            y_f = predict_first_spike(ids_o, t_o, Np)
            y_r = predict_race_to_N(ids_o, t_o, Np, args.race_N)
            train_correct_c += int(y_c==label); train_correct_w += int(y_w==label)
            train_correct_f += int(y_f==label); train_correct_r += int(y_r==label)
            train_total += 1

            if args.norm_h:
                for j in range(args.hidden):
                    idx_j = np.where(S_in_h.j[:]==j)[0]
                    s = float(np.sum(S_in_h.w[idx_j]))
                    if s>0: S_in_h.w[idx_j] = S_in_h.w[idx_j] / s

        # Rebuild output weights from hidden preference
        update_out_weights_from_hidden_pref(hid_c0, hid_c1)

        acc_c = train_correct_c/max(1,train_total)
        acc_w = train_correct_w/max(1,train_total)
        acc_f = train_correct_f/max(1,train_total)
        acc_r = train_correct_r/max(1,train_total)

        # Validation (freeze STDP)
        val_correct_c = val_correct_w = val_correct_f = val_correct_r = 0
        val_total = 0
        for iv in val_idx:
            binv = X[iv]; lblv = int(y[iv])
            t0 = set_input_spikes(binv)
            G_h.v = 0; G_h.ge=0; G_h.gi=0
            G_out.v = 0; G_out.ge=0; G_out.gi=0
            S_in_h.eta = 0.0
            net.run(Tdur_ms * ms, report=None)
            masko = (M_o.t/ms >= t0) & (M_o.t/ms < t0 + Tdur_ms)
            ids_o = np.array(M_o.i[masko], dtype=int)
            t_o = np.array((M_o.t[masko]/ms) - t0, dtype=float)
            counts = counts_by_class(t0, t0+Tdur_ms)
            y_c = predict_count(counts)
            y_w = predict_weighted(ids_o, t_o, Np, args.readout_tau)
            y_f = predict_first_spike(ids_o, t_o, Np)
            y_r = predict_race_to_N(ids_o, t_o, Np, args.race_N)
            val_correct_c += int(y_c==lblv); val_correct_w += int(y_w==lblv)
            val_correct_f += int(y_f==lblv); val_correct_r += int(y_r==lblv)
            val_total += 1

        print(f"[Epoch {ep+1}/{args.epochs}] Train acc | count: {acc_c:.3f} | weighted: {acc_w:.3f} | first: {acc_f:.3f} | race-N: {acc_r:.3f}")
        print(f"                     [Val] acc | count: {val_correct_c/max(1,val_total):.3f} | weighted: {val_correct_w/max(1,val_total):.3f} | first: {val_correct_f/max(1,val_total):.3f} | race-N: {val_correct_r/max(1,val_total):.3f} on {val_total} samples")

    # TEST
    test_correct_c = test_correct_w = test_correct_f = test_correct_r = 0
    test_total = 0
    for it in test_idx:
        bint = X[it]; lblt = int(y[it])
        t0 = set_input_spikes(bint)
        G_h.v=0; G_h.ge=0; G_h.gi=0
        G_out.v=0; G_out.ge=0; G_out.gi=0
        S_in_h.eta = 0.0
        net.run(Tdur_ms * ms, report=None)
        masko = (M_o.t/ms >= t0) & (M_o.t/ms < t0 + Tdur_ms)
        ids_o = np.array(M_o.i[masko], dtype=int)
        t_o = np.array((M_o.t[masko]/ms) - t0, dtype=float)
        counts = counts_by_class(t0, t0+Tdur_ms)
        y_c = predict_count(counts)
        y_w = predict_weighted(ids_o, t_o, Np, args.readout_tau)
        y_f = predict_first_spike(ids_o, t_o, Np)
        y_r = predict_race_to_N(ids_o, t_o, Np, args.race_N)
        test_correct_c += int(y_c==lblt); test_correct_w += int(y_w==lblt)
        test_correct_f += int(y_f==lblt); test_correct_r += int(y_r==lblt)
        test_total += 1

    print(f"[TEST] acc | count: {test_correct_c/max(1,test_total):.3f} | weighted: {test_correct_w/max(1,test_total):.3f} | first: {test_correct_f/max(1,test_total):.3f} | race-N: {test_correct_r/max(1,test_total):.3f} on {test_total} samples")

# -------------------------
# Main
# -------------------------

def main():
    args = parse_args()
    X, y, C, T = load_npz_folder(args.npz_dir, args.x_key, args.label_key,
                                 C=args.channels, T=args.time, strict=args.strict_shape)
    print(f"Loaded {len(X)} samples, shape=({C},{T}), labels 0/1")
    prefs.codegen.target = 'numpy'

    train_and_eval(X, y, C, T, args)

if __name__ == "__main__":
    main()
