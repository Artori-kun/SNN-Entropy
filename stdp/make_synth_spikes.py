#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_synth_spikes_refractory.py
-------------------------------
Synthetic spike dataset with two classes:
- Class 0: independent Poisson-like spikes (Bernoulli per bin with rate0)
- Class 1: Bursty spikes on a fixed set of "signature" channels.
           Inside each burst window, spikes respect a *refractory* period
           (no two spikes are closer than 'refractory' bins). Between bursts,
           background noise spikes are added.

Each sample saved as .npz with keys {x_key: (C,T) uint8, label_key: 0/1}.
Also writes meta.json describing the configuration and signature channels.

Example:
python make_synth_spikes_refractory.py --outdir spike_synth/ch32   --channels 32 --time 100 --n-train 320 --n-val 80 --n-test 100   --rate0 0.02 --burst-ch 8 --n-bursts 3 --burst-width 6   --high-rate 0.6 --refractory 2 --noise-rate 0.02   --jitter 2 --sync 0.7 --seed 42
"""

import json, argparse, math
from pathlib import Path
import numpy as np

def bernoulli_poisson(C, T, rate, rng):
    """Binary (C,T) with independent spikes: P(1)=rate per bin."""
    return (rng.random((C, T)) < rate).astype(np.uint8)

def gen_class0_poisson(C, T, rate0, rng):
    return bernoulli_poisson(C, T, rate0, rng)

def draw_burst_with_refractory(L, p, refractory, rng):
    """
    Return a length-L 0/1 vector where accepted spikes have min spacing 'refractory' bins.
    Algorithm: scan left->right; at each step, accept with prob p, then skip 'refractory' bins.
    Note: refractory=0 reduces to Bernoulli(p) per bin.
    Approx expected spikes per burst ~ L / (refractory + 1/p).
    """
    if L <= 0: return np.zeros(L, dtype=np.uint8)
    refractory = max(0, int(refractory))
    out = np.zeros(L, dtype=np.uint8)
    t = 0
    while t < L:
        if rng.random() < p:
            out[t] = 1
            t += refractory + 1
        else:
            t += 1
    return out

def gen_class1_bursty(C, T, sig_channels, n_bursts, burst_width,
                      high_rate, refractory, noise_rate, jitter, sync, rng):
    """
    sig_channels: indices of channels that carry bursts.
    n_bursts: number of bursts per sample (expected).
    burst_width: window length (bins) for each burst.
    high_rate: acceptance probability inside burst when scanning.
    refractory: min spacing (bins) between spikes within a burst.
    noise_rate: background spike probability outside bursts (applied to all channels).
    jitter: max +/- (bins) channel-wise shift of burst centers.
    sync: 0..1 probability to use the same burst center as the global one (else jittered).
    """
    X = bernoulli_poisson(C, T, noise_rate, rng)  # background noise for all channels
    sig_mask = np.zeros(C, dtype=bool)
    sig_mask[np.asarray(sig_channels, dtype=int)] = True

    if n_bursts <= 0 or burst_width <= 0:
        return X

    margin = max(1, int(math.ceil(burst_width/2)))
    valid_ts = np.arange(margin, T - margin) if T > 2*margin else np.arange(T)
    if valid_ts.size == 0:
        centers = np.array([], dtype=int)
    else:
        centers = np.sort(rng.choice(valid_ts, size=min(n_bursts, valid_ts.size), replace=False))

    for ch in np.where(sig_mask)[0]:
        for c in centers:
            center = c if (rng.random() <= sync) else int(np.clip(c + rng.integers(-jitter, jitter+1), 0, T-1))
            start = int(max(0, center - burst_width//2))
            end   = int(min(T, start + burst_width))
            L = end - start
            if L <= 0: continue
            burst_vec = draw_burst_with_refractory(L, high_rate, refractory, rng)
            # write burst over noise (OR)
            X[ch, start:end] = np.maximum(X[ch, start:end], burst_vec.astype(np.uint8))
    return X

def gen_dataset(outdir, C, T, n_train, n_val, n_test,
                rate0, burst_ch, n_bursts, burst_width,
                high_rate, refractory, noise_rate, jitter, sync,
                x_key="x", label_key="label", seed=0):
    rng = np.random.default_rng(seed)
    outdir = Path(outdir)
    (outdir / "train").mkdir(parents=True, exist_ok=True)
    (outdir / "val").mkdir(parents=True, exist_ok=True)
    (outdir / "test").mkdir(parents=True, exist_ok=True)

    # Fixed signature channels for the whole dataset
    sig_channels = sorted(rng.choice(np.arange(C), size=min(burst_ch, C), replace=False).tolist())

    meta = {
        "channels": C, "time": T,
        "train": n_train, "val": n_val, "test": n_test,
        "class0": {"type": "poisson", "rate0": rate0},
        "class1": {"type": "bursty", "sig_channels": sig_channels, "n_bursts": n_bursts,
                   "burst_width": burst_width, "high_rate": high_rate, "refractory": refractory,
                   "noise_rate": noise_rate, "jitter": jitter, "sync": sync},
        "x_key": x_key, "label_key": label_key, "seed": seed
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def save_split(split, n):
        n0 = n//2; n1 = n - n0
        for i in range(n0):
            X = gen_class0_poisson(C, T, rate0, rng)
            path = outdir / split / f"{split}_c0_{i:05d}.npz"
            np.savez_compressed(path, **{x_key: X.astype(np.uint8), label_key: np.uint8(0)})
        for i in range(n1):
            X = gen_class1_bursty(C, T, sig_channels, n_bursts, burst_width,
                                  high_rate, refractory, noise_rate, jitter, sync, rng)
            path = outdir / split / f"{split}_c1_{i:05d}.npz"
            np.savez_compressed(path, **{x_key: X.astype(np.uint8), label_key: np.uint8(1)})

    save_split("train", n_train)
    save_split("val",   n_val)
    save_split("test",  n_test)
    print(f"[OK] Wrote dataset to: {outdir}")
    print(f"Signature channels (class 1 bursts): {sig_channels}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, help="Output dir (creates train/val/test)")
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--time", type=int, default=100)
    ap.add_argument("--n-train", type=int, default=320)
    ap.add_argument("--n-val", type=int, default=80)
    ap.add_argument("--n-test", type=int, default=100)
    # class 0
    ap.add_argument("--rate0", type=float, default=0.02, help="P(spike)/bin for class 0 (Poisson)")
    # class 1 (bursty)
    ap.add_argument("--burst-ch", type=int, default=8, help="number of signature channels with bursts")
    ap.add_argument("--n-bursts", type=int, default=3, help="bursts per sample")
    ap.add_argument("--burst-width", type=int, default=6, help="burst width (bins)")
    ap.add_argument("--high-rate", type=float, default=0.6, help="accept prob inside burst (before refractory)")
    ap.add_argument("--refractory", type=int, default=2, help="min spacing between spikes within a burst (bins)")
    ap.add_argument("--noise-rate", type=float, default=0.02, help="background P(spike)/bin outside bursts")
    ap.add_argument("--jitter", type=int, default=2, help="+/- bins timing jitter between channels")
    ap.add_argument("--sync", type=float, default=0.7, help="0..1 prob to use shared burst centers across channels")
    # I/O
    ap.add_argument("--x-key", type=str, default="x")
    ap.add_argument("--label-key", type=str, default="label")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    gen_dataset(args.outdir, args.channels, args.time,
                args.n_train, args.n_val, args.n_test,
                args.rate0, args.burst_ch, args.n_bursts, args.burst_width,
                args.high_rate, args.refractory, args.noise_rate, args.jitter, args.sync,
                x_key=args.x_key, label_key=args.label_key, seed=args.seed)

if __name__ == "__main__":
    main()
