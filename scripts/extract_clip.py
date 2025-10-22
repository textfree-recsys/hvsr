#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
from PIL import Image
from io import BytesIO

import torch
import open_clip
from tqdm import tqdm

def find_image_key(example_dict):
    keys = list(example_dict.keys())
    # Common keys in DVBPR dumps: 'imgs'
    for k in keys:
        if 'img' in k.lower():
            return k
    raise KeyError(f"No image-like key found in item keys: {keys}")

def load_items(npy_path):
    arr = np.load(npy_path, allow_pickle=True, encoding='latin1')
    # Expected structure (6-tuple): user_train, user_val, user_test, Items, user_count, item_count
    if isinstance(arr, np.ndarray) and arr.shape == ():
        obj = arr.item()
        # If it's a dict-like, try to find the items
        if 'Items' in obj:
            items = obj['Items']
        else:
            # Fallback: assume the fourth entry is Items
            # but dict doesn't guarantee order. We warn.
            raise ValueError("Unsupported npy structure (dict) without 'Items' key.")
    else:
        # Assume tuple-like
        try:
            _, _, _, items, _, _ = arr
        except Exception:
            raise ValueError("Unsupported npy structure; expected 6-tuple with Items at index 3.")
    return items

def get_device(name):
    if name == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU", file=sys.stderr)
        return 'cpu'
    return name

def main():
    ap = argparse.ArgumentParser(description="Extract frozen CLIP image embeddings from a DVBPR .npy file")
    ap.add_argument("--npy", required=True, help="Path to DVBPR .npy file")
    ap.add_argument("--out", required=True, help="Output .npy for CLIP embeddings")
    ap.add_argument("--model", default="ViT-B-32", help="CLIP model (e.g., ViT-B-32, ViT-B-16)")
    ap.add_argument("--pretrained", default="openai", help="Pretrained weights identifier (e.g., openai)")
    ap.add_argument("--batch", type=int, default=256, help="Batch size")
    ap.add_argument("--device", default="cuda", choices=["cuda","cpu"], help="Device")
    ap.add_argument("--normalize", action="store_true", help="L2-normalize output embeddings")
    args = ap.parse_args()

    device = get_device(args.device)

    # Load items
    items = load_items(args.npy)
    if isinstance(items, np.ndarray):
        # np array of dicts
        first = items[0]
    elif isinstance(items, list):
        first = items[0]
    else:
        # could be dict -> values()
        first = list(items)[0] if isinstance(items, dict) else None

    if first is None or not isinstance(first, dict):
        raise ValueError("Unsupported Items structure. Expected array/list of dicts with image bytes.")

    img_key = find_image_key(first)

    # Load CLIP
    print(f"Loading CLIP: model={args.model} pretrained={args.pretrained}")
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained, device=device)
    model.eval()

    # Iterate items in batches
    n_items = len(items)
    feats = None

    def to_image_bytes(val):
        # Items may store JPEG bytes as bytes or Latin-1 encoded string
        if isinstance(val, bytes):
            return val
        if isinstance(val, str):
            return val.encode('latin1')
        # Some dumps might store as numpy bytes_
        if isinstance(val, np.bytes_):
            return bytes(val)
        raise TypeError(f"Unsupported image storage type: {type(val)}")

    with torch.no_grad():
        out_list = []
        for start in tqdm(range(0, n_items, args.batch), desc="Encoding"):
            end = min(start + args.batch, n_items)
            batch_imgs = []
            for i in range(start, end):
                b = to_image_bytes(items[i][img_key])
                img = Image.open(BytesIO(b)).convert("RGB")
                batch_imgs.append(preprocess(img))

            batch_tensor = torch.stack(batch_imgs, dim=0).to(device, non_blocking=True)
            emb = model.encode_image(batch_tensor)
            if args.normalize:
                emb = torch.nn.functional.normalize(emb, dim=-1)
            out_list.append(emb.cpu())

        feats = torch.cat(out_list, dim=0).float().numpy()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, feats)
    print(f"Saved CLIP embeddings: {feats.shape} -> {args.out}")

if __name__ == "__main__":
    main()
