#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Brian2 R-STDP (population outputs) with modulatory teacher, continuous eligibility,
# GAP-gated learning (use only eligibility accrued during GAP), optional margin-scaled
# teacher strength, and weight decay.

import argparse, json
from pathlib import Path
import numpy as np
from brian2 import (ms, prefs, defaultclock,
                    SpikeGeneratorGroup, NeuronGroup, Synapses, Network,
                    SpikeMonitor, clip)

def parse_args():
    p = argparse.ArgumentParser(description="R-STDP (pop outputs, GAP-gated) for NPZ spikes")
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
    p.add_argument("--seed", type=int, default=42)
    # Learning / traces
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--Apre", type=float, default=1.0)
    p.add_argument("--Apost", type=float, default=1.0)
    p.add_argument("--tau_pre", type=float, default=70.0)
    p.add_argument("--tau_post", type=float, default=120.0)
    p.add_argument("--tau_e", type=float, default=600.0)
    p.add_argument("--k_elig", type=float, default=3.0)
    p.add_argument("--wmax", type=float, default=2.0)
    p.add_argument("--decay", type=float, default=0.0, help="Weight decay per update, e.g., 0.001")
    p.add_argument("--norm_per_class", action="store_true")
    # Neuron dynamics
    p.add_argument("--out_pop", type=int, default=8, help="Neurons per class (total=2*out_pop)")
    p.add_argument("--v_th", type=float, default=0.55)
    p.add_argument("--tau_mem", type=float, default=20.0)
    p.add_argument("--tau_syn", type=float, default=12.0)
    p.add_argument("--refractory", type=float, default=5.0)
    p.add_argument("--inh_w", type=float, default=2.0)
    p.add_argument("--inh_tau", type=float, default=10.0)
    p.add_argument("--i_bias", type=float, default=0.05, help="Bias current to outputs")
    # Scheduling
    p.add_argument("--train_split", type=float, default=0.8)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--val_log", action="store_true")
    p.add_argument("--gap", type=float, default=25.0)
    # Teacher (modulatory in GAP; boosts postsyn ytrace only)
    p.add_argument("--teacher", action="store_true")
    p.add_argument("--teacher_trace_w", type=float, default=1.6)
    p.add_argument("--teacher_nspike", type=int, default=14)
    # Reward and margin scaling
    p.add_argument("--anti_hebbian", action="store_true")
    p.add_argument("--reward_beta", type=float, default=3.0, help="Margin shaping factor for DA")
    p.add_argument("--margin_da", action="store_true", help="Scale teacher strength by |tanh(beta*margin)|")
    # Homeostasis
    p.add_argument("--target_spk", type=float, default=3.0)
    p.add_argument("--th_adapt", type=float, default=0.02)
    # Output
    p.add_argument("--report", type=str, default="./rstdp_report")
    return p.parse_args()

def load_npz_folder(folder, x_key="x", label_key="label", C=None, T=None, strict=False):
    folder = Path(folder); files = sorted(folder.glob("*.npz"))
    if not files: raise FileNotFoundError(f"No .npz files in {folder}")
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
    defaultclock.dt = dt_ms * ms
    Np = params['out_pop']; Nout = 2 * Np

    G_in = SpikeGeneratorGroup(C, indices=np.array([], dtype=int), times=np.array([])*ms, name='G_in')

    eqs = '''
    dv/dt = (-v + ge - gi + I0) / tau_m : 1 (unless refractory)
    dge/dt = -ge / tau_e : 1
    dgi/dt = -gi / tau_i : 1
    dytrace/dt = -ytrace / tau_post_ng : 1
    tau_m : second
    tau_e : second
    tau_i : second
    tau_post_ng : second
    v_th : 1
    I0 : 1
    Apost_out : 1
    '''
    G_out = NeuronGroup(Nout, eqs, threshold='v>v_th',
                        reset='v=0; ytrace += Apost_out',
                        refractory=params['refractory']*ms, method='euler', name='G_out')
    G_out.tau_m = params['tau_mem'] * ms
    G_out.tau_e = params['tau_syn'] * ms
    G_out.tau_i = params['inh_tau'] * ms
    G_out.tau_post_ng = params['tau_post'] * ms
    G_out.v_th = params['v_th']; G_out.Apost_out = params['Apost']
    G_out.I0 = params['i_bias']; G_out.v = 0

    # Input->Output synapses with continuous eligibility
    S = Synapses(G_in, G_out,
                 model='''
                 w : 1
                 dxpre/dt = -xpre/taupre : 1 (clock-driven)
                 delig/dt = (-elig + k_elig * xpre * ytrace_post) / tau_elig : 1 (clock-driven)
                 taupre : second
                 tau_elig : second
                 k_elig : 1
                 Apre : 1
                 ''',
                 on_pre='''
                 ge_post += w
                 xpre += Apre
                 ''', name='S')
    S.connect(True)
    rng = np.random.default_rng(params['seed'])
    S.w = rng.uniform(0.0, 0.3, size=len(S))
    S.taupre = params['tau_pre'] * ms
    S.tau_elig = params['tau_e'] * ms
    S.k_elig = params['k_elig']
    S.Apre = params['Apre']

    # Lateral inhibition
    S_inh = Synapses(G_out, G_out, on_pre='gi_post += w_inh', model='w_inh:1', name='S_inh')
    S_inh.connect(condition='i!=j'); S_inh.w_inh = params['inh_w']

    # Modulatory teacher: one-to-one with output neurons (size = Nout)
    G_teacher = SpikeGeneratorGroup(Nout, indices=np.array([], dtype=int), times=np.array([])*ms, name='G_teacher')
    S_teach = Synapses(G_teacher, G_out, on_pre='ytrace_post += w_teach', model='w_teach:1', name='S_teach')
    S_teach.connect(j='i'); S_teach.w_teach = params.get('teacher_trace_w', 1.0)

    M_out = SpikeMonitor(G_out, name='M_out')
    net = Network(G_in, G_out, S, S_inh, G_teacher, S_teach, M_out)
    return net, G_in, G_out, S, M_out, G_teacher, S_teach, Np

def train_and_eval(X, y, C, T, args):
    dt_ms = float(args.dt); Tdur_ms = T * dt_ms; gap_ms = float(args.gap)
    rng = np.random.default_rng(args.seed)
    n = len(X); n_train = int(np.floor(args.train_split * n))
    idx = np.arange(n); rng.shuffle(idx)
    n_val = int(np.floor(args.val_split * n))
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]

    params = dict(seed=args.seed, tau_mem=args.tau_mem, tau_syn=args.tau_syn, inh_tau=args.inh_tau,
                  refractory=args.refractory, tau_pre=args.tau_pre, tau_post=args.tau_post, tau_e=args.tau_e,
                  k_elig=args.k_elig, Apre=args.Apre, Apost=args.Apost, lr=args.lr, wmax=args.wmax,
                  v_th=args.v_th, inh_w=args.inh_w, norm_per_class=args.norm_per_class,
                  teacher_trace_w=args.teacher_trace_w, out_pop=args.out_pop, i_bias=args.i_bias)
    net, G_in, G_out, S, M_out, G_teacher, S_teach, Np = build_network(C, dt_ms, params)
    Nout = 2 * Np

    def normalize_per_class():
        if not args.norm_per_class: return
        for cls in range(2):
            for j in range(cls*Np, (cls+1)*Np):
                mask = np.where(S.j[:] == j)[0]
                s = float(np.sum(S.w[mask]))
                if s > 0: S.w[mask] = S.w[mask] / s

    def class_counts(t0, t1):
        counts = [0, 0]
        for cls in range(2):
            j_start, j_end = cls*Np, (cls+1)*Np
            mask_time = (M_out.t/ms >= t0) & (M_out.t/ms < t1)
            mask_cls = np.isin(M_out.i, np.arange(j_start, j_end))
            counts[cls] = int(np.sum(mask_time & mask_cls))
        return counts

    train_acc_hist = []
    for ep in range(args.epochs):
        rng.shuffle(train_idx); correct = 0; total = 0
        W0 = S.w[:].copy(); elig_abs_gap_acc, elig_cnt = 0.0, 0

        for i in train_idx:
            binmat, label = X[i], int(y[i])
            t_now = float(defaultclock.t/ms)
            inds, times_ms = spikes_from_matrix(binmat, dt_ms)
            G_in.set_spikes(inds, (t_now + times_ms) * ms)

            # Reset state
            G_out.v = 0; G_out.ge = 0; G_out.gi = 0; G_out.ytrace = 0

            # Run TRIAL only
            t_start = t_now; t_end = t_now + Tdur_ms
            net.run(Tdur_ms * ms, report=None)

            # Read train prediction from trial window
            counts = class_counts(t_start, t_end)
            y_pred = int(np.argmax(counts)); correct += int(y_pred == label); total += 1

            # Margin for DA scaling (if enabled)
            diff = counts[label] - counts[1-label]; total_spk = max(1, counts[0] + counts[1])
            margin = diff / total_spk
            R_mag = float(np.tanh(args.reward_beta * margin))  # [-1,1]

            # Snapshot eligibility at end of trial
            elig_end_trial = S.elig[:].copy()

            # Schedule TEACHER in GAP (modulatory only) to the entire correct class population
            if args.teacher:
                j_start, j_end = label*Np, (label+1)*Np
                js = np.arange(j_start, j_end, dtype=int)
                times = np.linspace(t_end, t_end + gap_ms, num=args.teacher_nspike, endpoint=False)
                idxs = np.repeat(js, len(times))
                times_rep = np.tile(times, len(js))
                # Optionally scale teacher synaptic strength by |R_mag| for the labeled class
                if args.margin_da:
                    # Reset to baseline, then scale for labeled class
                    S_teach.w_teach = args.teacher_trace_w
                    mask = (S_teach.j[:] >= j_start) & (S_teach.j[:] < j_end)
                    S_teach.w_teach[mask] = args.teacher_trace_w * abs(R_mag)
                G_teacher.set_spikes(idxs, times_rep * ms)
            else:
                G_teacher.set_spikes(np.array([], dtype=int), np.array([])*ms)

            # Run GAP only
            net.run(gap_ms * ms, report=None)

            # GAP-only eligibility contribution
            e_gap = S.elig[:] - elig_end_trial
            elig_abs_gap_acc += float(np.mean(np.abs(e_gap))); elig_cnt += 1

            # Label-coded reward vector
            r_vec = np.array([0.0, 0.0])
            r_vec[label] = 1.0
            if args.anti_hebbian:
                r_vec[1-label] = -1.0
            mod = r_vec[(S.j[:] // Np)]

            # Weight decay + update with GAP eligibility only
            if args.decay > 0.0:
                S.w = S.w * (1.0 - args.decay)
            S.w = S.w + args.lr * (e_gap * mod)
            S.w = clip(S.w, 0, args.wmax)
            S.elig = 0 * S.elig
            normalize_per_class()

            # Homeostasis on trial spikes
            for j in range(Nout):
                nspk = int(np.sum((M_out.t/ms >= t_start) & (M_out.t/ms < t_end) & (M_out.i == j)))
                G_out.v_th[j] += args.th_adapt * (nspk - args.target_spk)

        train_acc = correct / max(1, total)
        dW_mean = float(np.mean(np.abs(S.w[:] - W0)))
        elig_gap_mean = (elig_abs_gap_acc/elig_cnt) if elig_cnt>0 else 0.0
        train_acc_hist.append(train_acc)
        print(f"[Epoch {ep+1}/{args.epochs}] Train acc: {train_acc:.3f} | Δw̄: {dW_mean:.4f} | |elig_gap|̄: {elig_gap_mean:.4f}")

        # ----- Validation (no teacher, no learning) -----
        val_correct = 0; val_total = 0
        val_dir = Path(args.report) / "val_logs" / f"epoch_{ep+1:03d}"
        if args.val_log:
            val_dir.mkdir(parents=True, exist_ok=True)
        for i_val in val_idx:
            binmat_v, label_v = X[i_val], int(y[i_val])
            t_now = float(defaultclock.t/ms)
            inds_v, times_v = spikes_from_matrix(binmat_v, dt_ms)
            G_in.set_spikes(inds_v, (t_now + times_v) * ms)
            # No teacher during validation
            G_teacher.set_spikes(np.array([], dtype=int), np.array([])*ms)
            # Reset state
            G_out.v = 0; G_out.ge = 0; G_out.gi = 0; G_out.ytrace = 0
            t_start = t_now; t_end = t_now + Tdur_ms
            net.run(Tdur_ms * ms, report=None)
            counts_v = class_counts(t_start, t_end)
            y_pred_v = int(np.argmax(counts_v))
            val_correct += int(y_pred_v == label_v); val_total += 1
            if args.val_log:
                mask_time = (M_out.t/ms >= t_start) & (M_out.t/ms < t_end)
                out_neuron = np.array(M_out.i[mask_time], dtype=int)
                out_t_ms = np.array((M_out.t[mask_time]/ms) - t_start, dtype=float)
                np.savez_compressed(val_dir / f"sample_{int(i_val):04d}.npz",
                                    out_neuron=out_neuron, out_t_ms=out_t_ms,
                                    label=label_v, counts=np.array(counts_v, dtype=int))
            # Advance by GAP to keep global time consistent
            net.run(gap_ms * ms, report=None)

        val_acc = val_correct / max(1, val_total)
        print(f"           [Val] acc: {val_acc:.3f} on {val_total} samples")
    
    # TEST (no teacher)
    correct = 0; total = 0
    for i in test_idx:
        binmat, label = X[i], int(y[i])
        t_now = float(defaultclock.t/ms)
        inds, times_ms = spikes_from_matrix(binmat, dt_ms)
        G_in.set_spikes(inds, (t_now + times_ms) * ms)
        G_teacher.set_spikes(np.array([], dtype=int), np.array([])*ms)
        G_out.v = 0; G_out.ge = 0; G_out.gi = 0; G_out.ytrace = 0
        t_start = t_now; t_end = t_now + Tdur_ms
        net.run(Tdur_ms * ms, report=None)
        counts = class_counts(t_start, t_end)
        y_pred = int(np.argmax(counts)); correct += int(y_pred == label); total += 1
        # advance by GAP to keep timing consistent
        net.run(gap_ms * ms, report=None)

    test_acc = correct / max(1, total)
    print(f"[TEST] Accuracy: {test_acc:.3f} on {total} samples")
    return {"train_acc_last": float(train_acc_hist[-1]) if train_acc_hist else None,
            "test_acc": float(test_acc), "n_train": int(len(train_idx)), "n_test": int(len(test_idx))}, S

def main():
    args = parse_args(); out_dir = Path(args.report); out_dir.mkdir(parents=True, exist_ok=True)
    X, y, C, T = load_npz_folder(args.npz_dir, args.x_key, args.label_key,
                                 C=args.channels, T=args.time, strict=args.strict_shape)
    print(f"Loaded {len(X)} samples, shape = ({C},{T}), labels 0/1")
    prefs.codegen.target = 'numpy'
    metrics, S = train_and_eval(X, y, C, T, args)
    np.save(out_dir / "weights_final.npy", S.w[:])
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f: json.dump(metrics, f, indent=2)
    print("Saved:", (out_dir / "weights_final.npy").resolve())

if __name__ == "__main__":
    main()
