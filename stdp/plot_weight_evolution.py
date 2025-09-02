#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Plot weight evolution from weights_log.npz

import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="weights_log.npz")
    args = ap.parse_args()
    d = np.load(args.npz, allow_pickle=True)
    W0 = d["W0"]  # shape (E+1, C, Np)
    W1 = d["W1"]
    dmean = d["dmean"]; cos_prev = d["cos_prev"]; cos_first = d["cos_first"]
    sat_hi = d["sat_hi"]; sat_lo = d["sat_lo"]

    E = W0.shape[0]-1
    fig, ax = plt.subplots(2,2, figsize=(11,8))

    ax[0,0].plot(range(1,E+1), dmean, marker='o')
    ax[0,0].set_title("Mean |Δw| per epoch")
    ax[0,0].set_xlabel("epoch"); ax[0,0].set_ylabel("mean abs change")

    ax[0,1].plot(range(1,E+1), cos_prev, label="cos(prev)", marker='o')
    ax[0,1].plot(range(1,E+1), cos_first, label="cos(first)", marker='s', alpha=0.8)
    ax[0,1].set_title("Cosine similarity"); ax[0,1].legend(); ax[0,1].set_xlabel("epoch")

    ax[1,0].plot(range(1,E+1), sat_hi*100, label=">=0.95 wmax")
    ax[1,0].plot(range(1,E+1), sat_lo*100, label="<=1e-4", alpha=0.8)
    ax[1,0].set_title("Saturation fraction (%)"); ax[1,0].legend(); ax[1,0].set_xlabel("epoch")

    # Column L1 norms (heatmap) for W0+W1 combined
    L = []
    for e in range(E+1):
        L.append(np.concatenate([np.linalg.norm(W0[e],1,axis=0), np.linalg.norm(W1[e],1,axis=0)]))
    L = np.array(L)  # (E+1, 2*Np)
    im = ax[1,1].imshow(L.T, aspect='auto', origin='lower')
    ax[1,1].set_title("Column L1 norms (E-pop | E-pop)")
    ax[1,1].set_xlabel("epoch"); ax[1,1].set_ylabel("column index")
    fig.colorbar(im, ax=ax[1,1], shrink=0.8, pad=0.02)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
