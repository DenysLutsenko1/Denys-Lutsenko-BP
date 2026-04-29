from __future__ import annotations

import sys
import os
sys.path.append(os.path.expanduser("~/work/python_libs"))

import argparse, yaml, joblib, torch
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, f1_score
# Импортируем расширенную загрузку
from src.datasets import load_jsonl, texts_aug_images_labels 
from src.encoders import TextEncoder, ImageEncoder
from src.utils import get_device

# Импортируем ту же функцию фузии, что и в train_multi
from src.train_multi import fuse 

def main(cfg, ckpt, threshold: float):
    # 1. Загрузка данных (теперь с картинками и аугментациями)
    test = load_jsonl(cfg['paths']['test'])
    t_txt, t_aug, t_img_paths, yte = texts_aug_images_labels(test)

    device = get_device(cfg['encoder'].get('device','auto'))
    bundle = joblib.load(ckpt)

    # 2. Инициализация энкодеров из бандла
    txt_enc = TextEncoder(bundle['text']['encoder_name'], device, 
                          max_length=bundle['text']['max_length'], 
                          pool=bundle['text'].get('pool', 'mean'))
    img_enc = ImageEncoder(bundle['image']['encoder_name'], device)

    # 3. Кодирование (LaCLIP стиль)
    print("Encoding test data...")
    # На тесте можно использовать оригинал + аугментации для стабильности
    X_txt = txt_enc.encode_laclip(t_txt, t_aug)
    
    # Кодируем картинки (только те, что есть)
    imgs_to_proc = [p for p in t_img_paths if p]
    X_img_raw = img_enc.encode_paths(imgs_to_proc, batch_size=cfg['image_encoder']['batch_size'])

    # 4. Фузия (собираем вектор признаков так же, как при обучении)
    use_sim = bundle.get("multimodal", {}).get("use_similarity", True)
    X_fused = fuse(X_txt, X_img_raw, t_txt, t_img_paths, use_similarity=use_sim)

    # 5. Предсказание
    clf = bundle['pipeline']
    prob = clf.predict_proba(X_fused)[:, 1]
    pred = (prob >= threshold).astype(int)

    # Метрики
    print(f"\nTarget metrics (Threshold: {threshold}):")
    try:
        print("AUC:", roc_auc_score(yte, prob))
    except: pass
    print(classification_report(yte, pred, digits=4))

if __name__ == "__main__":
    # Аргументы остаются те же
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='config.yaml')
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--threshold', type=float, default=0.5)
    args = ap.parse_args()
    
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    main(cfg, args.ckpt, args.threshold)