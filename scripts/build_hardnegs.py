#!/usr/bin/env python3
import argparse
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors

def main():
    ap = argparse.ArgumentParser(description="Precompute CLIP-space kNN hard negatives (cosine)")
    ap.add_argument("--clip", required=True, help="Path to CLIP embeddings .npy (num_items x dim)")
    ap.add_argument("--k", type=int, default=20, help="Neighbors to keep (excluding self)")
    ap.add_argument("--out", required=True, help="Output .npy for neighbor indices")
    ap.add_argument("--metric", default="cosine", help="Distance metric (default: cosine)")
    args = ap.parse_args()

    emb = np.load(args.clip)
    if emb.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array, got shape {emb.shape}")

    n_neighbors = args.k + 1  # +1 to include self, which we'll drop
    print(f"Fitting NearestNeighbors on {emb.shape[0]} items, dim={emb.shape[1]}, metric={args.metric}")
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=args.metric, n_jobs=-1)
    nn.fit(emb)

    dists, nbrs = nn.kneighbors(emb, return_distance=True)
    # Exclude self in position 0
    nbrs = nbrs[:, 1:1+args.k]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, nbrs.astype(np.int32))
    print(f"Saved neighbor index matrix: {nbrs.shape} -> {args.out}")

if __name__ == "__main__":
    main()
