from __future__ import annotations
import sys
import os
import gc
import random
sys.path.append(os.path.expanduser("~/work/python_libs"))

import numpy as np
import torch
from rich import print


def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(pref: str = "auto") -> torch.device:
    if pref.startswith("cuda:"):
        idx = int(pref.split(":")[1])
        if torch.cuda.is_available():
            return torch.device(f"cuda:{idx}")
        raise RuntimeError(f"{pref} запрошена но CUDA недоступна!")

    if pref == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA недоступна!")

    if pref == "cpu":
        return torch.device("cpu")

    # auto
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[yellow]Warning: CUDA not found. BLIP-2 on CPU will be EXTREMELY slow![/yellow]")
    return device


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def log_vram():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / 1024**3
            res   = torch.cuda.memory_reserved(i)  / 1024**3
            print(f"[blue]GPU {i} — Allocated: {alloc:.2f} GB | Reserved: {res:.2f} GB[/blue]")


def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()