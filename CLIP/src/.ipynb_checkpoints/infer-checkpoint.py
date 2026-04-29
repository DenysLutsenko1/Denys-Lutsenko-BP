from __future__ import annotations

import sys
import os
sys.path.append(os.path.expanduser("~/work/python_libs"))


import argparse, joblib, yaml
from src.encoders import TextEncoder
from src.utils import get_device





def main(ckpt: str, texts: list[str]):
    bundle = joblib.load(ckpt)
    device = get_device('auto')
    enc = TextEncoder(bundle['encoder_name'], device, max_length=bundle['max_length'], pool=bundle['pool'])
    X = enc.encode(texts, batch_size=64)
    clf = bundle['pipeline']
    prob = clf.predict_proba(X)[:,1]
    for t, p in zip(texts, prob):
        print(f"p(genai)={p:.3f}\t{textwrap.shorten(t, width=100, placeholder='…')}")




if __name__ == "__main__":
    import textwrap
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--text', type=str, nargs='+', required=True)
    args = ap.parse_args()
    main(args.ckpt, args.text)