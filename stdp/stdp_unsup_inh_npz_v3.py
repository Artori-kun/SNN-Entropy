#!/usr/bin/env python3
# STDP (notebook architecture) **no teacher, no homeostasis**.
# - Input -> two excitatory output populations (G0, G1)
# - Pair-based STDP (trace form) on Input->Output
# - Lateral + cross inhibition via gi_post increments
# - Column-wise L1 normalization (w_norm) after each sample
# - **Unsupervised**: no fixed pop→class; infer mapping after each epoch from train subset (count decoder)
# - Count-only readout; logs per-epoch mean |Δw| and saturation
#
# Example:
# python stdp_notebook_no_teacher_nohomeo.py --npz-dir /path/to/ALL_NPZ \
#   --x-key spike_matrix --label-key label --epochs 15 --out-pop 8 \
#   --epoch-samples 128 --balance-train --balance-val --report
#
# Requires: brian2

import argparse, glob, json
from pathlib import Path
import numpy as np

from brian2 import (ms, defaultclock, start_scope,
                    NeuronGroup, SpikeGeneratorGroup, Synapses,
                    SpikeMonitor, Network)

EPS = 1e-9

# ---------------------------- Mapping & readout (count) ----------------------------
def winner_pop(ids, Np):
    if ids.size == 0: return 0
    return 0 if (ids < Np).sum() >= (ids >= Np).sum() else 1

def acc_count_with_mapping(events, labels, Np, mapping):
    y = np.asarray(labels, dtype=int)
    yh = []
    for ids, _t in events:
        ids = np.asarray(ids)
        wp = winner_pop(ids, Np)  # 0 or 1
        yh.append(mapping[wp])
    yh = np.asarray(yh, dtype=int)
    return float((yh == y).mean())

def infer_mapping_from_train(events, labels, Np):
    acc_A = acc_count_with_mapping(events, labels, Np, mapping=[0,1])
    acc_B = acc_count_with_mapping(events, labels, Np, mapping=[1,0])
    if acc_B > acc_A:
        return [1,0], acc_B, acc_A
    return [0,1], acc_A, acc_B

# ---------------------------- Data ----------------------------
def load_all_npz(dirpath, x_key="spike_matrix", label_key="label"):
    files = sorted(glob.glob(str(Path(dirpath)/"*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz in {dirpath}")
    X=[]; y=[]
    for f in files:
        d = np.load(f)
        X.append(d[x_key])
        y.append(int(d[label_key]))
    X=np.asarray(X,np.uint8); y=np.asarray(y,int)
    return X,y

def stratified_split_indices(y, val_split=0.2, test_split=0.0, seed=0):
    rng=np.random.default_rng(seed)
    idx=np.arange(len(y))
    tr=[]; va=[]; te=[]
    for c in np.unique(y):
        idc=idx[y==c]; rng.shuffle(idc)
        n=len(idc)
        nt=int(round(n*test_split))
        nv=int(round(n*val_split))
        te+=list(idc[:nt])
        va+=list(idc[nt:nt+nv])
        tr+=list(idc[nt+nv:])
    rng.shuffle(tr); rng.shuffle(va); rng.shuffle(te)
    return np.array(tr), np.array(va), np.array(te)

# ---------------------------- Network ----------------------------
def build_net(C, Np, params):
    start_scope()
    defaultclock.dt = params["dt_ms"]*ms

    G_in = SpikeGeneratorGroup(C, indices=np.array([],dtype=int), times=np.array([])*ms)

    # LIF (no teacher current, just bias)
    eqs = '''
    dv/dt = (-(v - v_rest) + ge - gi + I_bias)/tau_m : 1 (unless refractory)
    dge/dt = -ge/tau_e : 1
    dgi/dt = -gi/tau_i : 1
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
    for G in (G0,G1):
        G.v=params["v_reset"]
        G.v_rest=params["v_rest"]
        G.v_th=params["v_th"]
        G.v_reset=params["v_reset"]
        G.tau_m=params["tau_m"]*ms
        G.tau_e=params["tau_e"]*ms
        G.tau_i=params["tau_i"]*ms
        G.I_bias=params["i_bias"]

    # Pair-based STDP (trace)
    stdp_model = '''
    w : 1
    Apre  : 1
    Apost : 1
    taupre  : second
    taupost : second
    dApre   : 1
    dApost  : 1
    wmax    : 1
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
    S0=Synapses(G_in,G0,model=stdp_model,on_pre=on_pre,on_post=on_post)
    S1=Synapses(G_in,G1,model=stdp_model,on_pre=on_pre,on_post=on_post)
    S0.connect(True); S1.connect(True)
    for S in (S0,S1):
        S.w = np.random.uniform(0.0, params["w_init"], size=len(S.w))
        S.dApre  = params["eta"]
        S.dApost = -params["eta"]*params["beta"]
        S.taupre  = params["tau_pre"]*ms
        S.taupost = params["tau_post"]*ms
        S.wmax    = params["wmax"]

    # Inhibition (lateral + optional cross)
    I0=I1=X01=X10=None
    if params["inh_w"]>0:
        I0=Synapses(G0,G0,model="w_inh:1",on_pre="gi_post += w_inh")
        I0.connect(condition="i!=j"); I0.w_inh=params["inh_w"]
        I1=Synapses(G1,G1,model="w_inh:1",on_pre="gi_post += w_inh")
        I1.connect(condition="i!=j"); I1.w_inh=params["inh_w"]
    if params["cross_inh_w"]>0:
        X01=Synapses(G0,G1,model="w_inh:1",on_pre="gi_post += w_inh")
        X01.connect(True); X01.w_inh=params["cross_inh_w"]
        X10=Synapses(G1,G0,model="w_inh:1",on_pre="gi_post += w_inh")
        X10.connect(True); X10.w_inh=params["cross_inh_w"]

    M0=SpikeMonitor(G0); M1=SpikeMonitor(G1)

    net=Network(G_in,G0,G1,S0,S1,M0,M1)
    if I0: net.add(I0,I1)
    if X01: net.add(X01,X10)
    return net,G_in,G0,G1,S0,S1,M0,M1

def normalize_columns(S, target_sum=1.0):
    if target_sum<=0: return
    w=S.w[:].copy(); j=S.j[:]
    n_post=int(j.max())+1 if j.size>0 else 0
    if n_post==0: return
    sums=np.zeros(n_post,float)
    for idx,jj in enumerate(j): sums[jj]+=w[idx]
    scale=np.ones_like(sums)
    nz=sums>0
    scale[nz]=target_sum/sums[nz]
    for idx,jj in enumerate(j):
        w[idx]=np.clip(w[idx]*scale[jj],0.0,float(S.wmax[0]))
    S.w[:]=w

def syn_to_matrix(S, C, Np):
    W = np.zeros((C, Np), dtype=float)
    i = np.asarray(S.i[:]); j = np.asarray(S.j[:]); w = np.asarray(S.w[:])
    if i.size: W[i, j] = w
    return W

# ---------------------------- Run one epoch ----------------------------
def run_epoch(net, X, y, G_in, G0, G1, S0, S1, M0, M1, params, subset=None):
    C, T = X.shape[1], X.shape[2]
    dt_ms=params["dt_ms"]
    events=[]; labels=[]
    no_spike=0; total_spikes=0
    idx = subset if subset is not None else np.arange(len(X),dtype=int)

    for k in idx:
        x=X[k]; label=int(y[k])
        for G in (G0,G1):
            G.v = params["v_reset"]; G.ge=0; G.gi=0

        ch,tbin=np.where(x>0)
        tms=(tbin.astype(float)*dt_ms)
        # schedule at absolute time (net.t offset)
        G_in.set_spikes(ch.astype(int), (tms + float(net.t/ms))*ms)

        net.run((T*dt_ms)*ms)

        # collect spikes within this sample window
        t0=float(net.t/ms) - T*dt_ms
        ids0 = M0.i[(M0.t/ms >= t0) & (M0.t/ms < t0 + T*dt_ms)]
        t0s  = M0.t[(M0.t/ms >= t0) & (M0.t/ms < t0 + T*dt_ms)]/ms
        ids1 = M1.i[(M1.t/ms >= t0) & (M1.t/ms < t0 + T*dt_ms)]
        t1s  = M1.t[(M1.t/ms >= t0) & (M1.t/ms < t0 + T*dt_ms)]/ms

        out_ids=np.concatenate([np.array(ids0), params["Np"]+np.array(ids1)]).astype(int)
        out_t  =np.concatenate([np.array(t0s)-t0, np.array(t1s)-t0]).astype(float)
        events.append((out_ids,out_t)); labels.append(label)

        total_spikes += out_ids.size
        if out_ids.size==0: no_spike+=1

        # column-wise L1 normalization
        normalize_columns(S0, target_sum=params["w_norm"])
        normalize_columns(S1, target_sum=params["w_norm"])

    diag = {"mean_spikes": total_spikes/float(len(idx)+EPS), "no_spike_pct": 100.0*no_spike/float(len(idx))}
    return events, labels, diag

# ---------------------------- Main ----------------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--x-key", default="spike_matrix")
    ap.add_argument("--label-key", default="label")
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

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

    # logging
    ap.add_argument("--save", type=str, default="stdp_no_teacher_nohomeo_log.npz")
    ap.add_argument("--report", action="store_true")
    args=ap.parse_args()

    rng=np.random.default_rng(args.seed)

    # load + split
    Xall,yall=load_all_npz(Path(args.npz_dir), x_key=args.x_key, label_key=args.label_key)
    itrv, ival, _=stratified_split_indices(yall, val_split=args.val_split, test_split=0.0, seed=args.seed)
    Xtr,ytr=Xall[itrv], yall[itrv]
    Xva,yva=Xall[ival], yall[ival]
    C,T=Xtr.shape[1], Xtr.shape[2]
    print(f"Loaded train {len(Xtr)}, val {len(Xva)} | shape=({C},{T}) | layout=single-folder")

    params=dict(
        dt_ms=args.dt_ms, v_th=args.v_th, v_rest=args.v_rest, v_reset=args.v_reset,
        tau_m=args.tau_m, tau_e=args.tau_e, tau_i=args.tau_i, t_ref=args.t_ref,
        i_bias=args.i_bias, eta=args.eta, beta=args.beta, tau_pre=args.tau_pre, tau_post=args.tau_post,
        wmax=args.wmax, w_init=args.w_init, w_norm=args.w_norm,
        inh_w=args.inh_w, cross_inh_w=args.cross_inh_w,
        Np=args.out_pop
    )

    net,G_in,G0,G1,S0,S1,M0,M1=build_net(C,args.out_pop,params)

    # weight logs
    W0_prev = syn_to_matrix(S0, C, args.out_pop)
    W1_prev = syn_to_matrix(S1, C, args.out_pop)
    dmean0_hist=[]; dmean1_hist=[]; dmean_hist=[]
    sat_hi_hist=[]; sat_lo_hist=[]
    mapping_hist=[]

    for ep in range(1, args.epochs+1):
        # subset (balanced if asked)
        n=min(args.epoch_samples,len(Xtr))
        if args.balance_train:
            id0=np.where(ytr==0)[0]; id1=np.where(ytr==1)[0]
            n0=n//2; n1=n-n0
            sel=np.concatenate([
                rng.choice(id0,size=min(n0,len(id0)),replace=len(id0)<n0),
                rng.choice(id1,size=min(n1,len(id1)),replace=len(id1)<n1),
            ]); rng.shuffle(sel)
        else:
            sel=rng.choice(np.arange(len(Xtr)),size=n,replace=False)

        # run epoch (unsupervised STDP)
        tr_events, tr_labels, tr_diag = run_epoch(net,Xtr,ytr,G_in,G0,G1,S0,S1,M0,M1,params,subset=sel)
        va_idx=np.arange(len(Xva),dtype=int)
        if args.balance_val:
            v0=np.where(yva==0)[0]; v1=np.where(yva==1)[0]
            m=min(len(v0),len(v1)); va_idx=np.concatenate([v0[:m],v1[:m]])
        va_events, va_labels, va_diag = run_epoch(net,Xva,yva,G_in,G0,G1,S0,S1,M0,M1,params,subset=va_idx)

        # ---- infer mapping (from train subset) ----
        mapping, tr_accA, tr_accB = infer_mapping_from_train(tr_events, tr_labels, args.out_pop)
        mapping_hist.append(mapping)

        # ---- compute acc using that mapping ----
        tr_acc = acc_count_with_mapping(tr_events, tr_labels, args.out_pop, mapping)
        va_acc = acc_count_with_mapping(va_events, va_labels, args.out_pop, mapping)

        # --- weight change
        W0 = syn_to_matrix(S0, C, args.out_pop)
        W1 = syn_to_matrix(S1, C, args.out_pop)
        d0 = np.mean(np.abs(W0 - W0_prev))
        d1 = np.mean(np.abs(W1 - W1_prev))
        d  = 0.5*(d0+d1)
        sat_hi = 0.5*(((W0 >= 0.95*args.wmax).mean()) + ((W1 >= 0.95*args.wmax).mean()))
        sat_lo = 0.5*(((W0 <= 1e-4).mean()) + ((W1 <= 1e-4).mean()))
        dmean0_hist.append(d0); dmean1_hist.append(d1); dmean_hist.append(d)
        sat_hi_hist.append(float(sat_hi)); sat_lo_hist.append(float(sat_lo))
        W0_prev, W1_prev = W0, W1

        if args.report:
            map_str = f"[map] pop0->class{mapping[0]}, pop1->class{mapping[1]} (train acc if fixed: A={tr_accA:.3f}, B={tr_accB:.3f})"
            print(f"[Epoch {ep:2d}/{args.epochs}] Train acc (mapped count): {tr_acc:.3f} | Val acc (mapped count): {va_acc:.3f}")
            print(f"                     {map_str}")
            print(f"           Δw̄0: {d0:.5f} | Δw̄1: {d1:.5f} | Δw̄: {d:.5f} | sat_hi: {sat_hi*100:.1f}% | sat_lo: {sat_lo*100:.1f}%")
            print(f"           [Diag] mean spk: {tr_diag['mean_spikes']:.1f} | no-spike%: {tr_diag['no_spike_pct']:.1f}% | "
                  f"val mean spk: {va_diag['mean_spikes']:.1f} | val no-spike%: {va_diag['no_spike_pct']:.1f}%")

    # save logs for later plotting
    np.savez(Path(args.save),
             dmean0=np.array(dmean0_hist),
             dmean1=np.array(dmean1_hist),
             dmean=np.array(dmean_hist),
             sat_hi=np.array(sat_hi_hist),
             sat_lo=np.array(sat_lo_hist),
             mapping=np.array(mapping_hist, dtype=int),
             meta=json.dumps(dict(vars(args))))
    print(f"[Saved] logs -> {args.save}")

if __name__ == '__main__':
    main()
