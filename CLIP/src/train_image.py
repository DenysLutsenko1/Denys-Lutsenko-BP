from __future__ import annotations

import sys
import os
sys.path.append(os.path.expanduser("~/work/python_libs"))


import os, argparse, yaml, joblib
import numpy as np
from rich import print
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from src.datasets import load_jsonl, texts_images_labels
from src.encoders import ImageEncoder
from src.models import make_clf
from src.utils import set_seed, get_device, ensure_dir

def main(cfg):
    print("[cyan]=== Train image-only ===[/cyan]")
    set_seed(cfg.get("random_seed", 42))

    train = load_jsonl(cfg["paths"]["train"])
    val   = load_jsonl(cfg["paths"]["val"])
    _, train_imgs, ytr = texts_images_labels(train)
    _, val_imgs,   yva = texts_images_labels(val)

    # фильтруем только те записи, где есть image_path
    tr_idx = [i for i, p in enumerate(train_imgs) if p]
    va_idx = [i for i, p in enumerate(val_imgs) if p]
    if len(tr_idx) == 0 or len(va_idx) == 0:
        raise RuntimeError("В train/val нет записей с image_path")

    train_imgs = [train_imgs[i] for i in tr_idx]
    ytr = [ytr[i] for i in tr_idx]
    val_imgs = [val_imgs[i] for i in va_idx]
    yva = [yva[i] for i in va_idx]

    device = get_device(cfg["image_encoder"].get("device","auto"))
    img_enc = ImageEncoder(cfg["image_encoder"]["name"], device)

    Xtr = img_enc.encode_paths(train_imgs, batch_size=cfg["image_encoder"]["batch_size"])
    Xva = img_enc.encode_paths(val_imgs,   batch_size=cfg["image_encoder"]["batch_size"])
    print(f"embeddings: Xtr={Xtr.shape}, Xva={Xva.shape}")

    clf = make_clf(cfg["train"]["clf_type"], normalize=cfg["features"]["normalize"],
                   lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"], epochs=cfg["train"]["epochs"])
    clf.fit(Xtr, ytr)

    prob = clf.predict_proba(Xva)[:,1]
    pred = (prob >= 0.5).astype(int)
    try:
        auc = roc_auc_score(yva, prob)
    except Exception as e:
        print("AUC error:", e); auc = None
    print("AUC:", auc)
    print("F1 :", f1_score(yva, pred))
    print(classification_report(yva, pred, digits=4))

    ensure_dir(cfg["paths"]["outdir"])
    outpath = os.path.join(cfg["paths"]["outdir"], "image_clf.joblib")
    joblib.dump({
        "pipeline": clf,
        "image_encoder_name": cfg["image_encoder"]["name"],
    }, outpath)
    print("[cyan]Saved ->[/cyan]", os.path.abspath(outpath))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
