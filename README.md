# Create a plain-text README file with simple formatting (no Markdown),
# based on the user's preferred content.

readme_txt = """\
Text-free visual–semantic recommendation via residual fusion and CLIP-guided negatives for cold-start
======================================================================================================

HVSR is a text-free multimodal recommender. It learns item embeddings from images only by:
- Extracting frozen CLIP image embeddings
- Learning visual features with a DVBPR-style CNN
- Fusing the two via a residual MLP into a 512-D item space
- Training with CLIP-space kNN hard negatives using a BPR-K objective

Why text-free?
Many catalogs lack reliable descriptions. HVSR works with just item images—while still leveraging CLIP’s semantic prior.

--------------------------------------------------------------------------------
Contents
--------------------------------------------------------------------------------
1) Highlights
2) Repository structure
3) Setup
4) Datasets
5) Extract CLIP embeddings
6) Precompute hard negatives
7) Train & evaluate (TensorFlow)
8) Quick sanity checks
9) Reproducibility notes
10) Acknowledgments
11) License
12) Citation
13) Contact

--------------------------------------------------------------------------------
1) Highlights
--------------------------------------------------------------------------------
- No text required at inference.
- Residual fusion for robust visual–semantic alignment.
- CLIP-guided hard negatives -> stronger gradients and better NDCG/Hit@K.
- Designed for cold-start and cross-domain scenarios (Amazon Fashion/Women/Men, Tradesy).

--------------------------------------------------------------------------------
2) Repository structure
--------------------------------------------------------------------------------
.
├─ main.py                    # CLI entry point (training & utilities)
├─ requirements.txt           # Python dependencies
├─ README.txt
├─ hvsr/
│  ├─ __init__.py
│  ├─ data/
│  │  └─ loaders.py          # DVBPR .npy loader utilities
│  ├─ models/
│  │  ├─ dvbpr_cnn_tf.py     # DVBPR-style CNN (TensorFlow)
│  │  ├─ fusion_tf.py        # Residual fusion block (TensorFlow)
│  │  └─ hvsr_tf.py          # Hybrid model wrapper (TensorFlow)
│  └─ train/
│     ├─ train_tf.py         # Training loop, eval, LR scheduler (TensorFlow)
│     └─ hardneg.py          # CLIP-space kNN with caching (scikit-learn)
├─ scripts/
│  ├─ download_dataset.sh    # Fetch DVBPR-formatted Amazon/Tradesy .npy files
│  ├─ extract_clip.py        # Compute frozen CLIP image embeddings (open-clip, PyTorch)
│  └─ build_hardnegs.py      # Precompute CLIP-space kNN matrix (optional helper)
└─ runs/                     # (created at runtime) logs, checkpoints, metrics

Note: We use PyTorch (open-clip) only once to extract frozen CLIP features. Training runs in TensorFlow.

--------------------------------------------------------------------------------
3) Setup
--------------------------------------------------------------------------------
Create and activate a virtual environment, then install dependencies.

Linux/macOS:
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt

(Optional) Check GPU visibility:
  python - <<'PY'
  import torch, tensorflow as tf
  print("torch cuda:", torch.cuda.is_available())
  print("tf gpus:", tf.config.list_physical_devices('GPU'))
  PY

--------------------------------------------------------------------------------
4) Datasets
--------------------------------------------------------------------------------
We use the DVBPR format: a single .npy containing
  user_train, user_val, user_test, Items, user_count, item_count
where Items[i]['imgs'] holds JPEG bytes for item i.

Download helper:
  bash scripts/download_dataset.sh --dataset all --out data
(Inside the script you can uncomment lines for Women/Men/Tradesy.)

--------------------------------------------------------------------------------
5) Extract CLIP embeddings
--------------------------------------------------------------------------------
Compute frozen CLIP (ViT-B/32) image embeddings once per dataset:

Example (Amazon Women):
  python scripts/extract_clip.py \
    --npy data/AmazonWomenWithImgPartitioned.npy \
    --out features/clip_embeddings_women.npy \
    --model ViT-B-32 \
    --batch 256 \
    --device cuda

Output shape is (num_items, 768) for ViT-B/32.
If you change the model, adjust clip-dim during training.

--------------------------------------------------------------------------------
6) Precompute hard negatives
--------------------------------------------------------------------------------
Build a CLIP-space kNN neighbor matrix (self excluded). This speeds up training.

Option A (via main.py):
  python main.py hardnegs \
    --clip features/clip_embeddings_women.npy \
    --k 20 \
    --out features/hard_negs_women_k20.npy

Option B (standalone helper):
  python scripts/build_hardnegs.py \
    --clip features/clip_embeddings_women.npy \
    --k 20 \
    --out features/hard_negs_women_k20.npy

--------------------------------------------------------------------------------
7) Train & evaluate (TensorFlow)
--------------------------------------------------------------------------------
Train the hybrid TensorFlow model (DVBPR-CNN + residual fusion) with BPR-K:

  python main.py train-tf \
    --npy data/AmazonWomenWithImgPartitioned.npy \
    --clip features/clip_embeddings_women.npy \
    --outdir runs/hvsr-women \
    --batch-size 512 \
    --epochs 50 \
    --lr 3e-4 \
    --num-neg 10 \
    --hard-k 20 \
    --cold-thresh 5 \
    --hard-cache features/hard_negs_women_k20.npy

Outputs:
- Checkpoints: runs/.../checkpoints/
- Metrics:     runs/.../metrics.json (AUC + cold AUC)
- Logs:        pipe stdout to a file if desired

--------------------------------------------------------------------------------
8) Quick sanity checks
--------------------------------------------------------------------------------
Inspect DVBPR split stats and cold threshold:
  python main.py dataset-stats \
    --npy data/AmazonWomenWithImgPartitioned.npy \
    --cold-thresh 5

Inspect CLIP embedding stats:
  python main.py clip-stats \
    --clip features/clip_embeddings_women.npy

--------------------------------------------------------------------------------
9) Reproducibility notes
--------------------------------------------------------------------------------
- CLIP is frozen to preserve general semantics and reduce compute.
- Fix random seeds (NumPy/TF) if you want stricter reproducibility.
- kNN uses scikit-learn NearestNeighbors(metric='cosine'). For very large catalogs consider FAISS IVFPQ.
- If you hit GPU OOM, reduce batch-size or num-neg.

--------------------------------------------------------------------------------
10) Acknowledgments
--------------------------------------------------------------------------------
- DVBPR data format/baselines: Kang et al., “Visually Aware Personalized Recommendation”
- CLIP image encoder: Radford et al., “Learning Transferable Visual Models From Natural Language Supervision”
- Thanks to the open-source community (TensorFlow, PyTorch, open-clip, scikit-learn).

--------------------------------------------------------------------------------
11) License
--------------------------------------------------------------------------------
Code: MIT License (see LICENSE).
Data: Follow original dataset licenses/terms (Amazon/Tradesy and DVBPR preprocessing).

--------------------------------------------------------------------------------
12) Citation
--------------------------------------------------------------------------------
If this project helps your research, please cite:

  Malhi, U. S., Siddeeq, S., Rasool, A., & Zhou, J. (2025).
  Text-free visual–semantic recommendation via residual fusion and CLIP-guided negatives for cold-start.
  Knowledge-Based Systems. Code: https://github.com/ORG/hvsr  DOI: 10.5281/zenodo.XXXXX

Tip: For a permanent reference, create a GitHub release/tag and connect the repo to Zenodo to mint a DOI.

--------------------------------------------------------------------------------
13) Contact
--------------------------------------------------------------------------------
Maintainer: Your Name
Email: you@example.com
GitHub: https://github.com/ORG/hvsr
"""

path = "/mnt/data/README_HVSR.txt"
with open(path, "w", encoding="utf-8") as f:
    f.write(readme_txt)

path
