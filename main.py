#!/usr/bin/env python3
"""
HVSR CLI

Subcommands:
  - train-tf       : Train the TensorFlow HVSR model (DVBPR CNN + residual fusion)
  - dataset-stats  : Quick stats from a DVBPR *WithImgPartitioned.npy
  - clip-stats     : Quick stats for a CLIP embeddings .npy (N,D)
  - hardnegs       : Precompute CLIP-space kNN hard negatives and save to .npy

Examples:

  python main.py train-tf \
      --npy data/AmazonWomenWithImgPartitioned.npy \
      --clip features/clip_embeddings_women.npy \
      --outdir runs/hvsr-women \
      --batch-size 512 --epochs 50 --lr 3e-4 \
      --num-neg 10 --hard-k 20 --cold-thresh 5

  python main.py dataset-stats --npy data/AmazonWomenWithImgPartitioned.npy
  python main.py clip-stats --clip features/clip_embeddings_women.npy
  python main.py hardnegs --clip features/clip_embeddings_women.npy --k 20 --out features/hard_negs.npy
"""

from __future__ import annotations
import argparse
import os
import sys
import numpy as np

__version__ = "0.1.0"

# ---- Safe imports for optional subcommands (avoid heavy deps at import time) ----
def _import_train_tf():
    try:
        from hvsr.train.train_tf import train_tf
        return train_tf
    except ImportError as e:
        # Give a helpful message if package import fails
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(here)
        try:
            from hvsr.train.train_tf import train_tf  # type: ignore
            return train_tf
        except Exception:
            raise ImportError(
                "Could not import hvsr.train.train_tf. "
                "Make sure your repo has 'hvsr/__init__.py' and 'hvsr/train/train_tf.py'."
            ) from e

def _import_hardneg():
    try:
        from hvsr.train.hardneg import build_or_load_hardnegs
        return build_or_load_hardnegs
    except ImportError as e:
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(here)
        try:
            from hvsr.train.hardneg import build_or_load_hardnegs  # type: ignore
            return build_or_load_hardnegs
        except Exception:
            raise ImportError(
                "Could not import hvsr.train.hardneg. "
                "Make sure your repo has 'hvsr/__init__.py' and 'hvsr/train/hardneg.py'."
            ) from e

# ---------------------------------------------------------------------------

def cmd_train_tf(args: argparse.Namespace) -> None:
    train_tf = _import_train_tf()
    train_tf(
        npy_path=args.npy,
        clip_path=args.clip,
        out_dir=args.outdir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        cold_thresh=args.cold_thresh,
        num_negatives=args.num_neg,
        hardneg_k=args.hard_k,
        hardneg_cache=args.hard_cache,
        clip_dim=args.clip_dim,
        cnn_feat_dim=args.cnn_dim,
        dropout=args.dropout,
    )

def cmd_dataset_stats(args: argparse.Namespace) -> None:
    arr = np.load(args.npy, allow_pickle=True, encoding="latin1")
    try:
        user_train, user_val, user_test, Items, user_count, item_count = arr
    except Exception:
        # Some DVBPR dumps may include different tuple packing:
        print("[error] Unexpected .npy structure. Expected 6-tuple (train,val,test,Items,usernum,itemnum).")
        raise

    user_count = int(user_count); item_count = int(item_count)
    print(f"[dataset] users={user_count} items={item_count}")
    print(f"[dataset] train users={len(user_train)} val users={len(user_val)} test users={len(user_test)}")

    # Peek item keys
    if item_count > 0:
        item0 = Items[0]
        if isinstance(item0, dict):
            print(f"[items] sample keys: {list(item0.keys())}")
        else:
            print("[items] Items[0] is not a dict; raw type:", type(item0))

    # Popularity rough stats
    key_candidates = ["productid", "itemid"]
    pid_key = None
    if item_count > 0 and isinstance(Items[0], dict):
        for k in key_candidates:
            if k in Items[0]:
                pid_key = k
                break
    pid_key = pid_key or "productid"

    pop = np.zeros(item_count, dtype=np.int32)
    for _, inters in user_train.items():
        for x in inters:
            idx = x.get(pid_key, x.get("itemid"))
            if idx is not None:
                pop[int(idx)] += 1

    cold_thresh = args.cold_thresh
    cold_cnt = int((pop < cold_thresh).sum())
    print(f"[dataset] cold items (<{cold_thresh} interactions): {cold_cnt}")

def cmd_clip_stats(args: argparse.Namespace) -> None:
    clip = np.load(args.clip)
    print(f"[clip] path={args.clip}")
    print(f"[clip] shape={clip.shape} dtype={clip.dtype}")
    if clip.ndim != 2:
        print("[warn] Expected CLIP embedding matrix with shape (N, D).")

def cmd_hardnegs(args: argparse.Namespace) -> None:
    build_or_load_hardnegs = _import_hardneg()
    clip = np.load(args.clip).astype(np.float32)
    out = args.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    hard = build_or_load_hardnegs(clip, k=args.k, cache_path=out)
    print(f"[hardnegs] saved: {out}  shape={hard.shape}")

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("hvsr", description="HVSR command line interface")
    p.add_argument("--version", action="version", version=f"HVSR {__version__}")

    sub = p.add_subparsers(dest="cmd", required=True)

    # --- train-tf ---
    p_tr = sub.add_parser("train-tf", help="Train HVSR (TensorFlow)")
    p_tr.add_argument("--npy", required=True, help="DVBPR *WithImgPartitioned.npy")
    p_tr.add_argument("--clip", required=True, help="CLIP embeddings .npy (N,D)")
    p_tr.add_argument("--outdir", default="runs/hvsr-tf", help="Output directory")
    p_tr.add_argument("--batch-size", type=int, default=512)
    p_tr.add_argument("--epochs", type=int, default=50)
    p_tr.add_argument("--lr", type=float, default=3e-4)
    p_tr.add_argument("--cold-thresh", type=int, default=5)
    p_tr.add_argument("--num-neg", type=int, default=10, help="Negatives per positive (BPR-K)")
    p_tr.add_argument("--hard-k", type=int, default=20, help="k-NN neighbors in CLIP space (exclude self)")
    p_tr.add_argument("--hard-cache", default=None, help="Path to save/load hard negatives .npy")
    p_tr.add_argument("--clip-dim", type=int, default=768)
    p_tr.add_argument("--cnn-dim", type=int, default=512)
    p_tr.add_argument("--dropout", type=float, default=0.5)
    p_tr.set_defaults(func=cmd_train_tf)

    # --- dataset-stats ---
    p_ds = sub.add_parser("dataset-stats", help="Print basic stats from DVBPR .npy")
    p_ds.add_argument("--npy", required=True, help="DVBPR *WithImgPartitioned.npy")
    p_ds.add_argument("--cold-thresh", type=int, default=5)
    p_ds.set_defaults(func=cmd_dataset_stats)

    # --- clip-stats ---
    p_cs = sub.add_parser("clip-stats", help="Print basic stats for CLIP embeddings .npy (N,D)")
    p_cs.add_argument("--clip", required=True)
    p_cs.set_defaults(func=cmd_clip_stats)

    # --- hardnegs ---
    p_hn = sub.add_parser("hardnegs", help="Precompute CLIP-space kNN hard negatives and save")
    p_hn.add_argument("--clip", required=True, help="CLIP embeddings .npy (N,D)")
    p_hn.add_argument("--k", type=int, default=20, help="Top-k neighbors (self excluded)")
    p_hn.add_argument("--out", required=True, help="Output .npy path (N,k)")
    p_hn.set_defaults(func=cmd_hardnegs)

    return p

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
        return 0
    except Exception as e:
        print(f"[error] {e.__class__.__name__}: {e}")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
