
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R-STDP (population outputs) with:
- Modulatory teacher in GAP, continuous eligibility (pre x post traces)
- Train/Val/Test splits (+ options to force class-balanced Val/Test)
- Balanced random train subset per epoch (optional)
- Spike-train logging (train/val/test)
- Multi-readout metrics per epoch: count, time-weighted, first-spike, race-to-N
- Validation info metrics on count-margin H(M), H(M|Y), MI(Y;M)

Prediction (population):
- Two output populations, each of size `out_pop` (total 2*out_pop).
- Vote by: total count / time-weighted count / first-spike / race-to-N.
- Count-margin M = (n1 - n0) / (n0 + n1), used for info metrics.
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
    p = argparse.ArgumentParser(description="R-STDP (pop outputs) with epoch subset, logging, multi-readouts, and info metrics")
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
    p.add_argument("--tau_pre", type=float, default=60.0)
    p.add_argument("--tau_post", type=float, default=60.0)  # postsynaptic ytrace tau (NeuronGroup)
    p.add_argument("--tau_e", type=float, default=400.0)
    p.add_argument("--k_elig", type=float, default=1.0)
    p.add_argument("--wmax", type=float, default=2.0)
    p.add_argument("--norm_per_class", action="store_true")

    # Neuron dynamics
    p.add_argument("--out_pop", type=int, default=8, help="Neurons per class (total = 2*out_pop)")
    p.add_argument("--v_th", type=float, default=0.6)
    p.add_argument("--tau_mem", type=float, default=20.0)
    p.add_argument("--tau_syn", type=float, default=10.0)
    p.add_argument("--refractory", type=float, default=5.0)
    p.add_argument("--inh_w", type=float, default=1.5)
    p.add_argument("--inh_tau", type=float, default=10.0)
    p.add_argument("--i_bias", type=float, default=0.0)

    # Scheduling
    p.add_argument("--train_split", type=float, default=0.8)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--gap", type=float, default=25.0)

    # Balancing options
    p.add_argument("--balance_val", action="store_true", help="Force validation set to be class-balanced")
    p.add_argument("--balance_test", action="store_true", help="Force test set to be class-balanced (from remaining after val)")

    # Teacher (modulatory in GAP; affects ytrace only)
    p.add_argument("--teacher", action="store_true")
    p.add_argument("--teacher_trace_w", type=float, default=1.0)
    p.add_argument("--teacher_start", type=float, default=None, help="Defaults to T (start in GAP)")
    p.add_argument("--teacher_end", type=float, default=None, help="Defaults to T+gap")
    p.add_argument("--teacher_nspike", type=int, default=12)

    # Reward shaping
    p.add_argument("--anti_hebbian", action="store_true")
    p.add_argument("--reward_beta", type=float, default=2.0)

    # Subset per epoch (balanced)
    p.add_argument("--epoch_samples", type=int, default=0,
                   help="If >0, number of train samples used each epoch (balanced per class). 0=use all.")

    # Spike logging
    p.add_argument("--train_log", action="store_true")
    p.add_argument("--val_log", action="store_true")
    p.add_argument("--test_log", action="store_true")

    # Readout options
    p.add_argument("--readout_tau", type=float, default=30.0, help="Tau (ms) for time-weighted count")
    p.add_argument("--race_N", type=int, default=3, help="Race-to-N spikes threshold")

    # Validation info metrics on count-margin
    p.add_argument("--val_info", action="store_true", help="Print H(M), H(M|Y), MI(Y;M) on validation (M = count-margin)")
    p.add_argument("--info_bins", type=int, default=21, help="Number of bins for margin histogram [-1,1]")

    p.add_argument("--report", type=str, default="./rstdp_report_subset")
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
# Stratified / balanced splits
# -------------------------

def make_balanced_splits(y, train_split, val_split, balance_val, balance_test, rng):
    n = len(y)
    idx0 = np.where(y==0)[0]
    idx1 = np.where(y==1)[0]
    rng.shuffle(idx0); rng.shuffle(idx1)

    # desired validation size
    n_val_total = int(np.floor(val_split * n))
    if balance_val:
        n_val_per = min(len(idx0), len(idx1), n_val_total//2)
        val_idx = np.concatenate([idx0[:n_val_per], idx1[:n_val_per]])
        idx0 = idx0[n_val_per:]; idx1 = idx1[n_val_per:]
    else:
        n_val0 = int(round(n_val_total * len(idx0)/(len(idx0)+len(idx1))))
        n_val1 = n_val_total - n_val0
        val_idx = np.concatenate([idx0[:n_val0], idx1[:n_val1]])
        idx0 = idx0[n_val0:]; idx1 = idx1[n_val1:]

    remaining = np.concatenate([idx0, idx1])
    rng.shuffle(remaining)

    n_train_total = int(np.floor(train_split * n))
    n_train_total = min(n_train_total, len(remaining))
    train_idx = remaining[:n_train_total]
    rest = remaining[n_train_total:]

    if balance_test and len(rest)>0:
        r0 = rest[y[rest]==0]; r1 = rest[y[rest]==1]
        rng.shuffle(r0); rng.shuffle(r1)
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
    # fallback to total counts
    return 0 if (out_ids < Np).sum() >= (out_ids >= Np).sum() else 1

# -------------------------
# Info metrics on margin M from counts
# -------------------------

def _entropy(p):
    p = p[np.isfinite(p) & (p>0)]
    return float(-np.sum(p * np.log2(p))) if p.size else 0.0

def info_from_counts_margins(n0_list, n1_list, y_list, B=21, eps=1e-12):
    n0 = np.asarray(n0_list, dtype=float)
    n1 = np.asarray(n1_list, dtype=float)
    y  = np.asarray(y_list,  dtype=int)
    m = (n1 - n0) / (n0 + n1 + eps)   # in [-1,1]
    bins = np.linspace(-1.0, 1.0, B+1)
    hist, _ = np.histogram(m, bins=bins); p = hist / max(1, hist.sum())
    H_M = _entropy(p)
    Hc = 0.0
    for c in (0,1):
        mc = m[y==c]
        if mc.size == 0: 
            continue
        hc, _ = np.histogram(mc, bins=bins); pc = hc / max(1, hc.sum())
        Hc += (mc.size/len(m)) * _entropy(pc)
    MI = H_M - Hc
    return H_M, Hc, MI

# -------------------------
# Network
# -------------------------

def build_network(C, dt_ms, params):
    Np = params['out_pop']; Nout = 2 * Np
    defaultclock.dt = dt_ms * ms

    # Input
    G_in = SpikeGeneratorGroup(C, indices=np.array([], dtype=int), times=np.array([])*ms, name='G_in')

    # Output neurons (LIF + ytrace), use tau_y (avoid '_post' naming conflict)
    eqs = '''
    dv/dt = (-v + ge - gi + Ibias) / tau_m : 1 (unless refractory)
    dge/dt = -ge / tau_e : 1
    dgi/dt = -gi / tau_i : 1
    dytrace/dt = -ytrace / tau_y : 1
    Ibias : 1
    tau_m : second
    tau_e : second
    tau_i : second
    tau_y : second
    v_th : 1
    Apost_out : 1
    '''
    G_out = NeuronGroup(2*Np, eqs,
                        threshold='v>v_th',
                        reset='v=0; ytrace += Apost_out',
                        refractory=params['refractory']*ms, method='euler',
                        name='outpop')
    G_out.tau_m = params['tau_mem'] * ms
    G_out.tau_e = params['tau_syn'] * ms
    G_out.tau_i = params['inh_tau'] * ms
    G_out.tau_y = params['tau_post'] * ms
    G_out.v_th = params['v_th']; G_out.Apost_out = params['Apost']; G_out.v = 0
    G_out.Ibias = params['i_bias']

    # In->Out synapses (eligibility, no '_post' variables defined here)
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
                 ''',
                 name='S_in_out')
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

    # Modulatory teacher (identity to outpop): bumps ytrace only
    G_teacher = SpikeGeneratorGroup(2*Np, indices=np.array([], dtype=int), times=np.array([])*ms, name='G_teacher')
    S_teach = Synapses(G_teacher, G_out, on_pre='ytrace_post += w_teach', model='w_teach:1', name='S_teach')
    S_teach.connect(j='i'); S_teach.w_teach = params['teacher_trace_w']

    # Monitors
    M_out = SpikeMonitor(G_out, name='M_out')

    net = Network(G_in, G_out, S, S_inh, G_teacher, S_teach, M_out)
    return net, G_in, G_out, S, M_out, G_teacher, Np

# -------------------------
# Train / Val / Test
# -------------------------

def train_and_eval(X, y, C, T, args):
    dt_ms = float(args.dt); Tdur_ms = T * dt_ms; gap_ms = float(args.gap)
    n = len(X)
    rng = np.random.default_rng(args.seed)

    train_idx, val_idx, test_idx = make_balanced_splits(y, args.train_split, args.val_split, args.balance_val, args.balance_test, rng)

    params = dict(seed=args.seed, tau_mem=args.tau_mem, tau_syn=args.tau_syn, inh_tau=args.inh_tau,
                  refractory=args.refractory, tau_pre=args.tau_pre, tau_post=args.tau_post, tau_e=args.tau_e,
                  k_elig=args.k_elig, Apre=args.Apre, Apost=args.Apost, lr=args.lr, wmax=args.wmax,
                  v_th=args.v_th, inh_w=args.inh_w, norm_per_class=args.norm_per_class,
                  teacher_trace_w=args.teacher_trace_w, out_pop=args.out_pop, i_bias=args.i_bias)
    net, G_in, G_out, S, M_out, G_teacher, Np = build_network(C, dt_ms, params)
    Nout = 2 * Np

    # Split summary
    def _sum(y_idx):
        if len(y_idx)==0: return (0,0,0.0)
        c0 = int(np.sum(y[y_idx]==0)); c1 = int(np.sum(y[y_idx]==1)); 
        ratio = c1 / max(1, c0 + c1)
        return c0, c1, ratio
    tr0,tr1,trr = _sum(train_idx); va0,va1,var = _sum(val_idx); te0,te1,ter = _sum(test_idx)
    print(f"[Split] Train: {len(train_idx)} (y0={tr0}, y1={tr1}, frac1={trr:.2f}) | Val: {len(val_idx)} (y0={va0}, y1={va1}, frac1={var:.2f}) | Test: {len(test_idx)} (y0={te0}, y1={te1}, frac1={ter:.2f})")

    def teacher_window_ms():
        start = Tdur_ms if args.teacher_start is None else float(args.teacher_start)
        end = Tdur_ms + gap_ms if args.teacher_end is None else float(args.teacher_end)
        return start, end

    def class_counts(t0, t1):
        mask = (M_out.t/ms >= t0) & (M_out.t/ms < t1)
        ids = np.array(M_out.i[mask], dtype=int)
        if ids.size == 0:
            return [0, 0]
        c0 = int(np.sum((ids >= 0) & (ids < Np)))
        c1 = int(np.sum((ids >= Np) & (ids < 2*Np)))
        return [c0, c1]

    def normalize_per_class():
        if not args.norm_per_class: return
        for j in range(Nout):
            idx_j = np.where(S.j[:] == j)[0]
            if idx_j.size > 0:
                s = float(np.sum(S.w[idx_j]))
                if s > 0: S.w[idx_j] = S.w[idx_j] / s

    def run_trial_and_log(binmat, label, do_teacher=False, log_spikes=False):
        t_now = float(defaultclock.t/ms)
        inds, times_ms = spikes_from_matrix(binmat, dt_ms)
        G_in.set_spikes(inds, (t_now + times_ms) * ms)

        # Teacher (GAP)
        if do_teacher and args.teacher:
            t0, t1 = teacher_window_ms()
            times_row = np.linspace(t_now + t0, t_now + t1, num=args.teacher_nspike).astype(float)
            group = np.arange(label*Np, (label+1)*Np, dtype=int)
            teach_idx = np.repeat(group, len(times_row))
            teach_times = np.tile(times_row, len(group))
            G_teacher.set_spikes(teach_idx, teach_times * ms)
        else:
            G_teacher.set_spikes(np.array([], dtype=int), np.array([])*ms)

        # Reset and run
        G_out.v = 0; G_out.ge = 0; G_out.gi = 0; G_out.ytrace = 0
        t_start = t_now; t_end = t_now + Tdur_ms
        net.run(Tdur_ms * ms, report=None)
        counts = class_counts(t_start, t_end)
        if gap_ms > 0:
            net.run(gap_ms * ms, report=None)

        logs = None
        if log_spikes:
            mask = (M_out.t/ms >= t_start) & (M_out.t/ms < t_end)
            out_ids = np.array(M_out.i[mask], dtype=int)
            out_t_rel = np.array((M_out.t[mask]/ms) - t_start, dtype=float)
            logs = (out_ids, out_t_rel, counts, label)
        return counts, logs

    # Metrics placeholders
    last_train_acc_count = last_train_acc_weight = last_train_acc_first = last_train_acc_race = None
    last_val_acc_count = last_val_acc_weight = last_val_acc_first = last_val_acc_race = None

    for ep in range(args.epochs):
        # Balanced subset
        if args.epoch_samples and args.epoch_samples > 0:
            idx0 = train_idx[y[train_idx] == 0]
            idx1 = train_idx[y[train_idx] == 1]
            m = min(args.epoch_samples//2, len(idx0), len(idx1))
            ep_idx = np.concatenate([rng.choice(idx0, m, replace=False),
                                     rng.choice(idx1, m, replace=False)])
            rng.shuffle(ep_idx)
        else:
            ep_idx = train_idx.copy(); rng.shuffle(ep_idx)

        # TRAIN
        train_correct_count = train_correct_weight = train_correct_first = train_correct_race = 0
        train_total = 0
        W_epoch_start = S.w[:].copy()
        train_dir = Path(args.report) / "train_logs" / f"epoch_{ep+1:03d}"
        if args.train_log: train_dir.mkdir(parents=True, exist_ok=True)

        for i in ep_idx:
            binmat = X[i]; label = int(y[i])
            counts, logs = run_trial_and_log(binmat, label, do_teacher=True, log_spikes=True)
            ids, t_rel = logs[0], logs[1]

            y_count  = predict_count(counts)
            y_weight = predict_weighted(ids, t_rel, Np, args.readout_tau)
            y_first  = predict_first_spike(ids, t_rel, Np)
            y_race   = predict_race_to_N(ids, t_rel, Np, args.race_N)

            train_correct_count  += int(y_count  == label)
            train_correct_weight += int(y_weight == label)
            train_correct_first  += int(y_first  == label)
            train_correct_race   += int(y_race   == label)
            train_total += 1

            # Reward
            diff = counts[label] - counts[1-label]; total_spk = max(1, counts[0] + counts[1])
            margin = diff / total_spk; R_mag = float(np.tanh(args.reward_beta * margin))
            r_vec = np.zeros(Nout, dtype=float)
            if args.anti_hebbian:
                r_vec[:] = -R_mag; r_vec[label*Np:(label+1)*Np] = +R_mag
            else:
                r_vec[label*Np:(label+1)*Np] = abs(R_mag)
            mod = r_vec[S.j[:]]

            # Update
            S.w = clip(S.w + args.lr * (S.elig * mod), 0, args.wmax)
            S.elig = 0 * S.elig
            normalize_per_class()

            if args.train_log and logs is not None:
                _, _, cts, lbl = logs
                np.savez_compressed(train_dir / f"sample_{int(i):04d}.npz",
                                    out_neuron=ids, out_t_ms=t_rel,
                                    label=lbl, counts=np.array(cts, dtype=int))

        acc_count  = train_correct_count / max(1, train_total)
        acc_weight = train_correct_weight / max(1, train_total)
        acc_first  = train_correct_first / max(1, train_total)
        acc_race   = train_correct_race / max(1, train_total)
        dW_mean = float(np.mean(np.abs(S.w[:] - W_epoch_start)))
        last_train_acc_count, last_train_acc_weight, last_train_acc_first, last_train_acc_race = acc_count, acc_weight, acc_first, acc_race
        print(f"[Epoch {ep+1}/{args.epochs}] Train acc | count: {acc_count:.3f} | weighted(tau={args.readout_tau:g}): {acc_weight:.3f} | first: {acc_first:.3f} | race-N(N={args.race_N}): {acc_race:.3f} | dW_mean: {dW_mean:.4f} | subset: {len(ep_idx)}")

        # VAL
        val_correct_count = val_correct_weight = val_correct_first = val_correct_race = 0
        val_total = 0
        _n0_list=[]; _n1_list=[]; _y_list=[]
        val_dir = Path(args.report) / "val_logs" / f"epoch_{ep+1:03d}"
        if args.val_log: val_dir.mkdir(parents=True, exist_ok=True)
        for i_v in val_idx:
            binmat_v = X[i_v]; label_v = int(y[i_v])
            counts_v, logs_v = run_trial_and_log(binmat_v, label_v, do_teacher=False, log_spikes=True)
            ids_v, t_rel_v = logs_v[0], logs_v[1]

            _n0_list.append(int(counts_v[0])); _n1_list.append(int(counts_v[1])); _y_list.append(label_v)

            y_count_v  = predict_count(counts_v)
            y_weight_v = predict_weighted(ids_v, t_rel_v, Np, args.readout_tau)
            y_first_v  = predict_first_spike(ids_v, t_rel_v, Np)
            y_race_v   = predict_race_to_N(ids_v, t_rel_v, Np, args.race_N)

            val_correct_count  += int(y_count_v  == label_v)
            val_correct_weight += int(y_weight_v == label_v)
            val_correct_first  += int(y_first_v  == label_v)
            val_correct_race   += int(y_race_v   == label_v)
            val_total += 1

            if args.val_log and logs_v is not None:
                _, _, cts, lbl = logs_v
                np.savez_compressed(val_dir / f"sample_{int(i_v):04d}.npz",
                                    out_neuron=ids_v, out_t_ms=t_rel_v,
                                    label=lbl, counts=np.array(cts, dtype=int))

        accc = val_correct_count / max(1, val_total)
        accw = val_correct_weight / max(1, val_total)
        accf = val_correct_first / max(1, val_total)
        accr = val_correct_race / max(1, val_total)
        last_val_acc_count, last_val_acc_weight, last_val_acc_first, last_val_acc_race = accc, accw, accf, accr

        # Info metrics H(M), H(M|Y), MI(Y;M) with M = (n1-n0)/(n0+n1)
        if args.val_info:
            H_M, Hc, MI = info_from_counts_margins(_n0_list, _n1_list, _y_list, B=args.info_bins)
            info_str = f" | Info(count-margin) H(M): {H_M:.4f} H(M|Y): {Hc:.4f} MI: {MI:.4f}"
        else:
            info_str = ""

        print(f"           [Val] acc | count: {accc:.3f} | weighted(tau={args.readout_tau:g}): {accw:.3f} | first: {accf:.3f} | race-N(N={args.race_N}): {accr:.3f} on {val_total} samples" + info_str)

    # TEST
    test_correct_count = test_correct_weight = test_correct_first = test_correct_race = 0
    test_total = 0
    test_dir = Path(args.report) / "test_logs"
    if args.test_log: test_dir.mkdir(parents=True, exist_ok=True)
    for i_t in test_idx:
        binmat_t = X[i_t]; label_t = int(y[i_t])
        counts_t, logs_t = run_trial_and_log(binmat_t, label_t, do_teacher=False, log_spikes=True)
        ids_t, t_rel_t = logs_t[0], logs_t[1]

        y_count_t  = predict_count(counts_t)
        y_weight_t = predict_weighted(ids_t, t_rel_t, Np, args.readout_tau)
        y_first_t  = predict_first_spike(ids_t, t_rel_t, Np)
        y_race_t   = predict_race_to_N(ids_t, t_rel_t, Np, args.race_N)

        test_correct_count  += int(y_count_t  == label_t)
        test_correct_weight += int(y_weight_t == label_t)
        test_correct_first  += int(y_first_t  == label_t)
        test_correct_race   += int(y_race_t   == label_t)
        test_total += 1

        if args.test_log and logs_t is not None:
            _, _, cts_t, lbl_t = logs_t
            np.savez_compressed(test_dir / f"sample_{int(i_t):04d}.npz",
                                out_neuron=ids_t, out_t_ms=t_rel_t,
                                label=lbl_t, counts=np.array(cts_t, dtype=int))

    accc_t = test_correct_count / max(1, test_total)
    accw_t = test_correct_weight / max(1, test_total)
    accf_t = test_correct_first / max(1, test_total)
    accr_t = test_correct_race / max(1, test_total)
    print(f"[TEST] acc | count: {accc_t:.3f} | weighted(tau={args.readout_tau:g}): {accw_t:.3f} | first: {accf_t:.3f} | race-N(N={args.race_N}): {accr_t:.3f} on {test_total} samples")

    return {
        "train_acc_last_count": float(last_train_acc_count) if last_train_acc_count is not None else None,
        "train_acc_last_weight": float(last_train_acc_weight) if last_train_acc_weight is not None else None,
        "train_acc_last_first": float(last_train_acc_first) if last_train_acc_first is not None else None,
        "train_acc_last_race": float(last_train_acc_race) if last_train_acc_race is not None else None,
        "val_acc_last_count": float(last_val_acc_count) if last_val_acc_count is not None else None,
        "val_acc_last_weight": float(last_val_acc_weight) if last_val_acc_weight is not None else None,
        "val_acc_last_first": float(last_val_acc_first) if last_val_acc_first is not None else None,
        "val_acc_last_race": float(last_val_acc_race) if last_val_acc_race is not None else None,
        "test_acc_count": float(accc_t),
        "test_acc_weight": float(accw_t),
        "test_acc_first": float(accf_t),
        "test_acc_race": float(accr_t),
        "n_train": int(len(train_idx)), "n_val": int(len(val_idx)), "n_test": int(len(test_idx))
    }, S

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
    metrics, S = train_and_eval(X, y, C, T, args)
    np.save(out_dir / "weights_final.npy", S.w[:])
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f: json.dump(metrics, f, indent=2)
    print("Saved:", (out_dir / "weights_final.npy").resolve())

if __name__ == "__main__":
    main()
