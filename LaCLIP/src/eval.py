from __future__ import annotations

import sys
import os
sys.path.append(os.path.expanduser("~/work/python_libs"))

import argparse
import yaml
import joblib
import torch
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from src.datasets import load_jsonl, texts_aug_images_labels 
from src.encoders import TextEncoder, ImageEncoder
from src.utils import get_device

# Импортируем ту же функцию фузии, что и в train_multi
from src.train_multi import fuse 

def main(cfg, ckpt, threshold: float):
    # 1. Загрузка данных
    print(f"Loading test data from: {cfg['paths']['test']}")
    test = load_jsonl(cfg['paths']['test'])
    t_txt, t_aug, t_img_paths, yte = texts_aug_images_labels(test)

    device = get_device(cfg['encoder'].get('device','auto'))
    print(f"Loading model bundle from: {ckpt}")
    bundle = joblib.load(ckpt)

    # 2. Инициализация энкодеров из бандла
    txt_enc = TextEncoder(
        bundle['text']['encoder_name'], 
        device, 
        max_length=bundle['text']['max_length'], 
        pool=bundle['text'].get('pool', 'mean')
    )
    img_enc = ImageEncoder(bundle['image']['encoder_name'], device)

    # 3. Кодирование
    print("Encoding test data (LaCLIP style)...")
    X_txt = txt_enc.encode_laclip(t_txt, t_aug)
    
    # Кодируем картинки (только те, что существуют)
    imgs_to_proc = [p for p in t_img_paths if p]
    X_img_raw = img_enc.encode_paths(imgs_to_proc, batch_size=cfg['image_encoder']['batch_size'])

    # 4. Фузия
    use_sim = bundle.get("multimodal", {}).get("use_similarity", True)
    X_fused = fuse(X_txt, X_img_raw, t_txt, t_img_paths, use_similarity=use_sim)

    # 5. Предсказание
    clf = bundle['pipeline']
    # Получаем вероятности для ROC AUC (только для положительного класса)
    prob = clf.predict_proba(X_fused)[:, 1]
    # Применяем порог для классификации
    pred = (prob >= threshold).astype(int)

    # --- РАСЧЕТ МЕТРИК ---
    try:
        roc_auc = roc_auc_score(yte, prob)
    except Exception:
        roc_auc = 0.0

    # Получаем отчет в виде словаря, чтобы вытащить общий F1
    report_dict = classification_report(yte, pred, output_dict=True)
    f1_weighted = report_dict['weighted avg']['f1-score']

    # --- КРАСИВЫЙ ВЫВОД КАК НА СКРИНШОТЕ ---
    print("\n" + "="*45)
    print("             TESTSET RESULTS             ")
    print("="*45)
    print(f"ROC AUC:  {roc_auc:.4f}")
    print(f"F1 Score: {f1_weighted:.4f}")
    print("-" * 45)
    
    # digits=4 обеспечивает точность как на образце (например, 0.9524)
    print(classification_report(yte, pred, digits=4))
    print("="*45)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='config.yaml')
    ap.add_argument('--ckpt', type=str, required=True, help="Path to your .joblib model")
    ap.add_argument('--threshold', type=float, default=0.5, help="Classification threshold")
    args = ap.parse_args()
    
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        main(cfg, args.ckpt, args.threshold)
    except FileNotFoundError:
        print(f"Error: Config file '{args.config}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")