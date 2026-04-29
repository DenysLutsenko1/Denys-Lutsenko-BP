from __future__ import annotations

import sys
import os
sys.path.append(os.path.expanduser("~/work/python_libs"))

import os, random
import numpy as np
import torch
from rich import print




def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)




def get_device(pref: str = "auto") -> torch.device:
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")




def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)