# Text-free visual–semantic recommendation via residual fusion and CLIP-guided negatives for cold-start

**HVSR** is a text-free multimodal recommender that learns item embeddings from **images only**. It:
- extracts **frozen CLIP** image embeddings,
- learns visual features with a **DVBPR-style CNN**,
- **fuses** both via a **residual MLP** into a 512-D space,
- trains with **CLIP-space kNN hard negatives** using a **BPR-K** objective.

> Why text-free? Many catalogs lack reliable descriptions. HVSR works with just item images while still leveraging CLIP’s semantic prior.

---

## Quick start

**1) Setup**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

(Optional) verify GPU visibility:
```bash
python - <<'PY'
import torch, tensorflow as tf
print("torch cuda:", torch.cuda.is_available())
print("tf gpus:", tf.config.list_physical_devices('GPU'))
PY
```

**2) Get data (DVBPR format)**
```bash
bash scripts/download_dataset.sh
# Produces e.g. data/AmazonWomenWithImgPartitioned.npy
```

**3) Extract frozen CLIP embeddings (once per dataset)**
```bash
python scripts/extract_clip.py   --npy data/AmazonWomenWithImgPartitioned.npy   --out features/clip_embeddings_women.npy   --model ViT-B-32   --batch 256   --device cuda
```

**4) Precompute CLIP-space hard negatives (optional but faster)**
```bash
# via main.py
python main.py hardnegs   --clip features/clip_embeddings_women.npy   --k 20   --out features/hard_negs_women_k20.npy

# or standalone helper
python scripts/build_hardnegs.py   --clip features/clip_embeddings_women.npy   --k 20   --out features/hard_negs_women_k20.npy
```

**5) Train (TensorFlow)**
```bash
python main.py train-tf   --npy data/AmazonWomenWithImgPartitioned.npy   --clip features/clip_embeddings_women.npy   --outdir runs/hvsr-women   --batch-size 512   --epochs 50   --lr 3e-4   --num-neg 10   --hard-k 20   --cold-thresh 5   --hard-cache features/hard_negs_women_k20.npy
```

**6) Sanity checks**
```bash
# dataset stats + cold threshold
python main.py dataset-stats --npy data/AmazonWomenWithImgPartitioned.npy --cold-thresh 5

# CLIP embedding stats
python main.py clip-stats --clip features/clip_embeddings_women.npy
```

---

## Repository layout
```
.
├─ main.py                    # CLI: training & utilities
├─ requirements.txt
├─ README.md
├─ hvsr/
│  ├─ __init__.py
│  ├─ data/
│  │  └─ loaders.py          # DVBPR .npy loader utilities
│  ├─ models/
│  │  ├─ dvbpr_cnn_tf.py     # DVBPR-style CNN (TensorFlow)
│  │  ├─ fusion_tf.py        # Residual fusion block (TensorFlow)
│  │  └─ hvsr_tf.py          # Hybrid model wrapper (TensorFlow)
│  └─ train/
│     ├─ train_tf.py         # Training loop, eval, LR scheduler
│     └─ hardneg.py          # CLIP-space kNN (scikit-learn)
├─ scripts/
│  ├─ download_dataset.sh    # Fetch DVBPR .npy files
│  ├─ extract_clip.py        # Frozen CLIP features (open-clip, PyTorch)
│  └─ build_hardnegs.py      # Precompute kNN neighbors
└─ runs/                     # (created at runtime) logs, ckpts, metrics
```
Note: PyTorch (open-clip) is used only to extract frozen CLIP features; training is in TensorFlow.

---

## Notes
- CLIP is **frozen** to keep general semantics and reduce compute.
- Set random seeds for stricter reproducibility (NumPy/TF).
- kNN uses `NearestNeighbors(metric='cosine')`. For very large catalogs, consider FAISS IVFPQ.
- If you see OOM, reduce `--batch-size` or `--num-neg`.

## Acknowledgments 
- DVBPR data format/baselines: Kang et al., “Visually-Aware Fashion Recommendation and
Design with Generative Image Models” 
- CLIP image encoder: Radford et al., “Learning Transferable Visual Models From Natural Language Supervision” 
- Thanks to the open-source community (TensorFlow, PyTorch, open-clip, scikit-learn).