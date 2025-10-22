import os
import requests
from dataclasses import dataclass
import numpy as np
from hvsr.utils.io import log, sizeof_fmt

DATA_URLS = {
    "fashion":  "http://cseweb.ucsd.edu/~wckang/DVBPR/AmazonFashion6ImgPartitioned.npy",
    "women":    "http://cseweb.ucsd.edu/~wckang/DVBPR/AmazonWomenWithImgPartitioned.npy",
    "men":      "http://cseweb.ucsd.edu/~wckang/DVBPR/AmazonMenWithImgPartitioned.npy",
    "tradesy":  "http://cseweb.ucsd.edu/~wckang/DVBPR/TradesyImgPartitioned.npy",
}

def _target_filename(name: str) -> str:
    return os.path.basename(DATA_URLS[name])

def download_dataset(name: str, out_dir: str = "data", overwrite: bool=False) -> str:
    if name not in DATA_URLS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(DATA_URLS)}")
    url = DATA_URLS[name]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, _target_filename(name))

    if os.path.exists(out_path) and not overwrite:
        log(f"[download] exists: {out_path} ({sizeof_fmt(os.path.getsize(out_path))}). Skipping.")
        return out_path

    log(f"[download] GET {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        chunk = 1024 * 1024
        downloaded = 0
        with open(out_path, "wb") as f:
            for b in r.iter_content(chunk_size=chunk):
                if b:
                    f.write(b)
                    downloaded += len(b)
                    if total:
                        pct = 100.0 * downloaded / total
                        print(f"\r{downloaded/(1024**2):.1f}MB / {total/(1024**2):.1f}MB ({pct:.1f}%)", end="")
    print()
    log(f"[download] saved to {out_path}")
    return out_path

@dataclass
class DVBPRDataset:
    user_train: object
    user_val: object
    user_test: object
    items: np.ndarray  # array of dicts; each has an 'imgs' field (bytes/str)
    usernum: int
    itemnum: int

def load_partitioned_npy(path: str) -> DVBPRDataset:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    arr = np.load(path, allow_pickle=True, encoding="latin1")
    user_train, user_val, user_test, Items, usernum, itemnum = arr
    return DVBPRDataset(
        user_train=user_train,
        user_val=user_val,
        user_test=user_test,
        items=Items,
        usernum=int(usernum),
        itemnum=int(itemnum),
    )