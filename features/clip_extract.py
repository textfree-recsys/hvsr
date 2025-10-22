import os
import math
import numpy as np
import torch
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModel
from hvsr.data.dvbpr import load_partitioned_npy
from hvsr.utils.imgs import decode_dvbpr_image
from hvsr.utils.io import ensure_dir, log, Timer, save_json

def extract_clip_embeddings(
    npy_path: str,
    out_path: str = "clip_embeddings.npy",
    model_name: str = "openai/clip-vit-base-patch32",
    batch_size: int = 256,
    device: str = "cuda",
    limit: int | None = None,
):
    """
    Load DVBPR .npy, decode item images, run CLIP vision encoder, save (N, D) embeddings.
    Returns metadata dict.
    """
    ds = load_partitioned_npy(npy_path)
    items = ds.items
    n_items = len(items) if limit is None else min(len(items), limit)
    ensure_dir(os.path.dirname(out_path) or ".")

    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    log(f"[clip] device={dev}")

    processor = CLIPImageProcessor.from_pretrained(model_name)
    model = CLIPVisionModel.from_pretrained(model_name)
    model.eval().to(dev)

    # Peek one item to get dim
    tmp_img = decode_dvbpr_image(items[0]["imgs"])
    tmp_inputs = processor(images=tmp_img, return_tensors="pt").to(dev)
    with torch.no_grad():
        tmp_out = model(**tmp_inputs)
        emb_dim = tmp_out.pooler_output.shape[-1]

    log(f"[clip] model={model_name}, embedding_dim={emb_dim}, items={n_items}, batch_size={batch_size}")

    num_batches = math.ceil(n_items / batch_size)
    out = np.memmap(out_path, dtype="float32", mode="w+", shape=(n_items, emb_dim))  # write-safe large file
    failed = 0

    with Timer("CLIP extraction"):
        idx = 0
        for b in tqdm(range(num_batches), desc="batches"):
            start, end = b * batch_size, min((b + 1) * batch_size, n_items)
            pil_batch = []
            for k in range(start, end):
                try:
                    pil_batch.append(decode_dvbpr_image(items[k]["imgs"]))
                except Exception:
                    # Fallback to a black image if decode fails
                    failed += 1
                    pil_batch.append(None)

            # Replace any None with a blank 224x224
            for i, img in enumerate(pil_batch):
                if img is None:
                    from PIL import Image
                    pil_batch[i] = Image.new("RGB", (224, 224), (0, 0, 0))

            inputs = processor(images=pil_batch, return_tensors="pt").to(dev)

            with torch.no_grad():
                outputs = model(**inputs)
                pooled = outputs.pooler_output  # (B, D)
                pooled = pooled.detach().cpu().numpy()

            out[idx:idx + len(pooled)] = pooled
            idx += len(pooled)

    # Flush memmap to disk
    out.flush()

    meta = {
        "npy_source": os.path.abspath(npy_path),
        "embeddings_path": os.path.abspath(out_path),
        "num_items": int(n_items),
        "embedding_dim": int(emb_dim),
        "model_name": model_name,
        "batch_size": int(batch_size),
        "device": str(dev),
        "failed_decodes": int(failed),
    }
    save_json(meta, out_path + ".meta.json")
    log(f"[clip] saved: {out_path} and {out_path}.meta.json")

    # Also return in-memory small dict
    return meta