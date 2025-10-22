import os
import json
import time

def log(msg: str):
    print(f"[hvsr] {msg}")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def sizeof_fmt(num, suffix="B"):
    for unit in ["","K","M","G","T","P","E","Z"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Y{suffix}"

class Timer:
    def __init__(self, tag=""):
        self.tag = tag
    def __enter__(self):
        self.t0 = time.time()
        return self
    def __exit__(self, *exc):
        dt = time.time() - self.t0
        log(f"{self.tag} took {dt:.2f}s")

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)