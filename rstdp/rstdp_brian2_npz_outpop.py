
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Brian2 R‑STDP with population output (outpop), validation set, and spike logging.

Data
----
Folder of .npz; each file is 1 sample with:
  - x: (C, T) binary spike matrix (e.g., (32,100))
  - label: scalar 0/1

Output population
-----------------
- Two classes (0/1), each represented by K neurons (K = --out_k).
- Total output neurons = 2*K. Indices:
    class 0: [0 .. K-1]
    class 1: [K .. 2K-1]
- Prediction uses sum of spikes within each class-group.

Learning
--------
- LIF output neurons with lateral inhibition.
- Pre-trace xpre (on pre spike), postsynaptic ytrace (on post spike), both decaying.
- Continuous eligibility per synapse: d(elig)/dt = (-elig + k_elig * xpre * ytrace_post)/tau_elig
- End-of-trial reward-modulated update (per-class, with optional anti-Hebbian on wrong class).
- Teacher is modulatory: only bumps ytrace_post (no ge) to avoid label leakage.
  Place teacher in GAP window to avoid affecting prediction.

Validation
----------
- After each epoch, run validation (no learning, no teacher) and print val acc.
- If --val_log is set, save per-sample spikes (input + output) to NPZ files under
  {REPORT}/val_logs/epoch_XXX/sample_####.npz

Requires: brian2, numpy
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
    p = argparse.ArgumentParser(description="Brian2 R‑STDP with population output + validation")
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

    # Output population
    p.add_argument("--out_k", type=int, default=8, help="Number of output neurons per class (>=1)")

    # Learning / traces
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--Apre", type=float, default=1.0)
    p.add_argument("--Apost", type=float, default=1.0)  # increment to postsynaptic ytrace on spike
    p.add_argument("--tau_pre", type=float, default=60.0)
    p.add_argument("--tau_post", type=float, default=60.0)  # postsynaptic ytrace tau
    p.add_argument("--tau_e", type=float, default=400.0)
    p.add_argument("--k_elig", type=float, default=1.0)
    p.add_argument("--wmax", type=float, default=2.0)
    p.add_argument("--norm_per_class", action="store_true",
                   help="L1-normalize incoming weights per postsyn neuron after each update")

    # Neuron dynamics
    p.add_argument("--v_th", type=float, default=0.6)
    p.add_argument("--tau_mem", type=float, default=20.0)
    p.add_argument("--tau_syn", type=float, default=10.0)
    p.add_argument("--refractory", type=float, default=5.0)
    p.add_argument("--inh_w", type=float, default=1.5)
    p.add_argument("--inh_tau", type=float, default=10.0)

    # Scheduling
    p.add_argument("--train_split", type=float, default=0.8)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--gap", type=float, default=25.0)

    # Teacher (modulatory only — bumps ytrace, not ge)
    p.add_argument("--teacher", action="store_true")
    p.add_argument("--teacher_trace_w", type=float, default=1.0,
                   help="Scale of teacher effect on postsynaptic ytrace")
    p.add_argument("--teacher_start", type=float, default=100.0, help="Start time within sample (ms)")
    p.add_argument("--teacher_end", type=float, default=120.0, help="End time within sample (ms)")
    p.add_argument("--teacher_nspike", type=int, default=12)

    # Reward shaping
    p.add_argument("--anti_hebbian", action="store_true")
    p.add_argument("--reward_beta", type=float, default=2.0)

    # Validation logging
    p.add_argument("--val_log", action="store_true")

    p.add_argument("--report", type=str, default="./rstdp_report")
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
# Network
# -------------------------

def build_network(C, dt_ms, params):
    K = params['out_k']
    Nout = 2 * K
    defaultclock.dt = dt_ms * ms

    # Input spikes
    G_in = SpikeGeneratorGroup(C, indices=np.array([], dtype=int), times=np.array([])*ms, name='G_in')

    # Output (Nout LIF neurons) with postsynaptic ytrace
    eqs = '''
    dv/dt = (-v + ge - gi) / tau_m : 1 (unless refractory)
    dge/dt = -ge / tau_e : 1
    dgi/dt = -gi / tau_i : 1
    dytrace/dt = -ytrace / tau_post_ng : 1
    tau_m : second
    tau_e : second
    tau_i : second
    tau_post_ng : second
    v_th : 1
    Apost_out : 1
    '''
    G_out = NeuronGroup(Nout, eqs, threshold='v>v_th', reset='v=0; ytrace += Apost_out',
                        refractory=params['refractory']*ms, method='euler', name='outpop')
    G_out.tau_m = params['tau_mem'] * ms
    G_out.tau_e = params['tau_syn'] * ms
    G_out.tau_i = params['inh_tau'] * ms
    G_out.tau_post_ng = params['tau_post'] * ms
    G_out.v_th = params['v_th']; G_out.Apost_out = params['Apost']; G_out.v = 0

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
                 ''', name='S_in_out')
    S.connect(True)
    rng = np.random.default_rng(params['seed'])
    S.w = rng.uniform(0.0, 0.3, size=len(S))
    S.taupre = params['tau_pre'] * ms
    S.tau_elig = params['tau_e'] * ms
    S.k_elig = params['k_elig']
    S.Apre = params['Apre']

    # Lateral inhibition (all-to-all except self)
    S_inh = Synapses(G_out, G_out, on_pre='gi_post += w_inh', model='w_inh:1', name='S_lateral_inh')
    S_inh.connect(condition='i!=j')
    S_inh.w_inh = params['inh_w']

    # Teacher population (modulatory only): Nout neurons, identity to outpop
    G_teacher = SpikeGeneratorGroup(Nout, indices=np.array([], dtype=int), times=np.array([])*ms, name='G_teacher')
    S_teach = Synapses(G_teacher, G_out, on_pre='ytrace_post += w_teach', model='w_teach:1', name='S_teacher')
    S_teach.connect(j='i')
    S_teach.w_teach = params['teacher_trace_w']

    # Monitors
    M_out = SpikeMonitor(G_out, name='M_out')

    net = Network(G_in, G_out, S, S_inh, G_teacher, S_teach, M_out)
    return net, G_in, G_out, S, M_out, G_teacher, K

# -------------------------
# Train / Val / Test
# -------------------------

def train_and_eval(X, y, C, T, args):
    dt_ms = float(args.dt); Tdur_ms = T * dt_ms; gap_ms = float(args.gap)
    n = len(X); n_train = int(np.floor(args.train_split * n)); n_val = int(np.floor(args.val_split * n))
    rng = np.random.default_rng(args.seed); idx = np.arange(n); rng.shuffle(idx)
    train_idx = idx[:n_train]; val_idx = idx[n_train:n_train+n_val]; test_idx = idx[n_train+n_val:]

    params = dict(seed=args.seed, tau_mem=args.tau_mem, tau_syn=args.tau_syn, inh_tau=args.inh_tau,
                  refractory=args.refractory, tau_pre=args.tau_pre, tau_post=args.tau_post, tau_e=args.tau_e,
                  Apre=args.Apre, Apost=args.Apost, lr=args.lr, wmax=args.wmax, v_th=args.v_th,
                  inh_w=args.inh_w, norm_per_class=args.norm_per_class, teacher_trace_w=args.teacher_trace_w,
                  k_elig=args.k_elig, out_k=args.out_k)
    net, G_in, G_out, S, M_out, G_teacher, K = build_network(C, dt_ms, params)

    def class_slice(c):
        # return slice of output indices for class c (0 or 1)
        start = c * K; return slice(start, start + K)

    def normalize_per_post():
        if not args.norm_per_class: return
        for j in range(len(G_out)):
            idx_j = np.where(S.j[:] == j)[0]
            if idx_j.size > 0:
                s = float(np.sum(S.w[idx_j]))
                if s > 0: S.w[idx_j] = S.w[idx_j] / s

    def run_trial_and_count(binmat, label, do_teacher=False, teacher_window=None, log_spikes=False):
        t_now = float(defaultclock.t/ms)
        inds, times_ms = spikes_from_matrix(binmat, dt_ms)
        G_in.set_spikes(inds, (t_now + times_ms) * ms)

        # Teacher spikes for the entire label group (modulatory only)
        if do_teacher and args.teacher:
            t0, t1 = teacher_window if teacher_window is not None else (args.teacher_start, args.teacher_end)
            times_row = np.linspace(t_now + t0, t_now + t1, num=args.teacher_nspike).astype(float)
            group = np.arange(class_slice(int(label)).start, class_slice(int(label)).stop, dtype=int)
            teach_idx = np.repeat(group, len(times_row))
            teach_times = np.tile(times_row, len(group))
            G_teacher.set_spikes(teach_idx, teach_times * ms)
        else:
            G_teacher.set_spikes(np.array([], dtype=int), np.array([])*ms)

        # Reset dynamics
        G_out.v = 0; G_out.ge = 0; G_out.gi = 0; G_out.ytrace = 0

        # Run and count in [t_start, t_end)
        t_start = t_now; t_end = t_now + Tdur_ms
        net.run((Tdur_ms + gap_ms) * ms, report=None)

        # Counts by group
        counts0 = int(np.sum((M_out.t/ms >= t_start) & (M_out.t/ms < t_end) & (M_out.i >= 0) & (M_out.i < K)))
        counts1 = int(np.sum((M_out.t/ms >= t_start) & (M_out.t/ms < t_end) & (M_out.i >= K) & (M_out.i < 2*K)))
        counts = [counts0, counts1]

        if not log_spikes: return counts, None

        # Log spikes relative to trial start
        in_idx = inds; in_t_rel = times_ms
        mask_win = (M_out.t/ms >= t_start) & (M_out.t/ms < t_end)
        out_neuron = np.array(M_out.i[mask_win], dtype=int)
        out_t_rel = np.array((M_out.t[mask_win]/ms) - t_start, dtype=float)
        logs = {"in_idx": in_idx.astype(int), "in_t_ms": in_t_rel.astype(float),
                "out_neuron": out_neuron, "out_t_ms": out_t_rel}
        return counts, logs

    train_acc_hist = []; val_hist = []
    for ep in range(args.epochs):
        rng.shuffle(train_idx); correct = 0; total = 0
        W_epoch_start = S.w[:].copy(); elig_abs_accum = 0.0; elig_count = 0

        for i in train_idx:
            binmat, label = X[i], int(y[i])
            counts, _ = run_trial_and_count(binmat, label, do_teacher=True,
                                            teacher_window=(args.teacher_start, args.teacher_end),
                                            log_spikes=False)
            y_pred = 0 if counts[0] >= counts[1] else 1
            correct += int(y_pred == label); total += 1

            # Margin-shaped reward per output neuron
            diff = counts[label] - counts[1-label]; total_spk = max(1, counts[0] + counts[1])
            margin = diff / total_spk; R_mag = float(np.tanh(args.reward_beta * margin))
            # Reward vector size Nout
            r_vec = np.zeros(2*K, dtype=float)
            if args.anti_hebbian:
                r_vec[:] = -R_mag; r_vec[class_slice(label)] = +R_mag
            else:
                r_vec[class_slice(label)] = abs(R_mag)
            mod = r_vec[S.j[:]]  # per-synapse post-based reward

            elig_abs_accum += float(np.mean(np.abs(S.elig[:]))); elig_count += 1

            # Update weights
            S.w = clip(S.w + args.lr * (S.elig * mod), 0, args.wmax); S.elig = 0 * S.elig
            normalize_per_post()

        train_acc = correct / max(1, total)
        dW_mean = float(np.mean(np.abs(S.w[:] - W_epoch_start)))
        elig_mean = (elig_abs_accum/elig_count) if elig_count>0 else 0.0
        train_acc_hist.append(train_acc)
        print(f"[Epoch {ep+1}/{args.epochs}] Train acc: {train_acc:.3f} | Δw̄: {dW_mean:.4f} | |elig|̄: {elig_mean:.4f}")

        # Validation
        val_correct = 0; val_total = 0
        val_dir = Path(args.report) / "val_logs" / f"epoch_{ep+1:03d}"
        if args.val_log: val_dir.mkdir(parents=True, exist_ok=True)
        for idx_v in val_idx:
            binmat_v, label_v = X[idx_v], int(y[idx_v])
            counts_v, logs = run_trial_and_count(binmat_v, label_v, do_teacher=False, teacher_window=None, log_spikes=args.val_log)
            y_pred_v = 0 if counts_v[0] >= counts_v[1] else 1
            val_correct += int(y_pred_v == label_v); val_total += 1
            if args.val_log:
                np.savez_compressed(val_dir / f"sample_{int(idx_v):04d}.npz",
                                    in_idx=logs["in_idx"], in_t_ms=logs["in_t_ms"],
                                    out_neuron=logs["out_neuron"], out_t_ms=logs["out_t_ms"],
                                    label=label_v, counts=np.array(counts_v, dtype=int))
        val_acc = val_correct / max(1, val_total); val_hist.append(val_acc)
        print(f"           [Val] acc: {val_acc:.3f} on {val_total} samples")

    # TEST (no learning/teacher)
    correct = 0; total = 0
    for i in test_idx:
        binmat, label = X[i], int(y[i])
        t_now = float(defaultclock.t/ms)
        inds, times_ms = spikes_from_matrix(binmat, dt_ms)
        G_in.set_spikes(inds, (t_now + times_ms) * ms)
        G_teacher.set_spikes(np.array([], dtype=int), np.array([])*ms)
        G_out.v = 0; G_out.ge = 0; G_out.gi = 0; G_out.ytrace = 0
        t_start = t_now; t_end = t_now + Tdur_ms
        net.run((Tdur_ms + gap_ms) * ms, report=None)
        counts0 = int(np.sum((M_out.t/ms >= t_start) & (M_out.t/ms < t_end) & (M_out.i >= 0) & (M_out.i < K)))
        counts1 = int(np.sum((M_out.t/ms >= t_start) & (M_out.t/ms < t_end) & (M_out.i >= K) & (M_out.i < 2*K)))
        y_pred = 0 if counts0 >= counts1 else 1
        correct += int(y_pred == label); total += 1
    test_acc = correct / max(1, total)
    print(f"[TEST] Accuracy: {test_acc:.3f} on {total} samples")

    return {"train_acc_last": float(train_acc_hist[-1]) if train_acc_hist else None,
            "val_acc_last": float(val_hist[-1]) if val_hist else None,
            "n_train": int(len(train_idx)), "n_val": int(len(val_idx)), "n_test": int(len(test_idx)),
            "test_acc": float(test_acc)}, S, K

# -------------------------
# Main
# -------------------------

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
    # Save mapping info
    mapping = {"out_k": int(K), "class0_indices": [int(i) for i in range(0, K)], "class1_indices": [int(i) for i in range(K, 2*K)]}
    with open(out_dir / "outpop_mapping.json", "w", encoding="utf-8") as f: json.dump(mapping, f, indent=2)
    print("Saved:", (out_dir / "weights_final.npy").resolve())
    print("Output population mapping saved to:", (out_dir / "outpop_mapping.json").resolve())

if __name__ == "__main__":
    main()
