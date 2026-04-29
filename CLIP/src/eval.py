# from __future__ import annotations

# import sys
# import os
# sys.path.append(os.path.expanduser("~/work/python_libs"))


# import argparse, yaml, joblib, os
# import numpy as np
# from sklearn.metrics import classification_report, roc_auc_score, f1_score

# # Импортируем те же функции, что использовали при обучении
# from src.datasets import load_jsonl, texts_images_labels
# from src.encoders import TextEncoder, ImageEncoder
# from src.utils import get_device

# def fuse_features(text_emb, img_emb, texts, imgs, use_similarity: bool = True):
#     """
#     Повторяем логику слияния из train_multi.py
#     """
#     Ht = text_emb.shape[1] if text_emb is not None else 0
#     Hi = img_emb.shape[1]  if img_emb  is not None else 0

#     feats = []
#     # Индексы картинок
#     img_positions = {idx: k for k, idx in enumerate([i for i, p in enumerate(imgs) if p])}

#     for i in range(len(texts)):
#         tvec = text_emb[i] if text_emb is not None else np.zeros(Ht, dtype=np.float32)
#         ivec = img_emb[img_positions[i]] if (imgs[i] and img_emb is not None) else np.zeros(Hi, dtype=np.float32)

#         parts = [tvec, ivec]

#         sim = 0.0
#         if use_similarity and Ht > 0 and Hi > 0 and (texts[i] and imgs[i]):
#             tv = tvec / (np.linalg.norm(tvec) + 1e-9)
#             iv = ivec / (np.linalg.norm(ivec) + 1e-9)
#             sim = float(np.dot(tv, iv))
#         parts.append(np.array([sim], dtype=np.float32))

#         feats.append(np.concatenate(parts, axis=0))

#     return np.vstack(feats)

# def main(cfg, ckpt, threshold: float):
#     # 1. Загрузка бандла и конфига
#     bundle = joblib.load(ckpt)
#     test_data = load_jsonl(cfg['paths']['test'])
#     # Используем мультимодальную загрузку (текст + фото + метка)
#     te_texts, te_imgs, yte = texts_images_labels(test_data)

#     device = get_device(cfg['encoder'].get('device', 'auto'))

#     # 2. Инициализация энкодеров (берем параметры из сохраненного бандла)
#     txt_enc = TextEncoder(
#         bundle['text']['encoder_name'], 
#         device, 
#         max_length=bundle['text']['max_length'], 
#         pool=bundle['text']['pool']
#     )
#     img_enc = ImageEncoder(bundle['image']['encoder_name'], device)

#     # 3. Кодирование
#     print("Encoding test data...")
#     # Кодируем тексты
#     Xte_txt = txt_enc.encode([t if isinstance(t, str) else "" for t in te_texts], 
#                              batch_size=cfg['encoder']['batch_size'])
    
#     # Кодируем картинки (только те, что существуют)
#     te_img_idx = [i for i, p in enumerate(te_imgs) if p]
#     Xte_img = img_enc.encode_paths([te_imgs[i] for i in te_img_idx], 
#                                    batch_size=cfg['image_encoder']['batch_size']) if te_img_idx else None

#     # 4. Слияние (Fusion)
#     use_sim = bundle.get('multimodal', {}).get('use_similarity', True)
#     Xte_fused = fuse_features(Xte_txt, Xte_img, te_texts, te_imgs, use_similarity=use_sim)

#     # 5. Классификация
#     clf = bundle['pipeline']
#     prob = clf.predict_proba(Xte_fused)[:, 1]
#     pred = (prob >= threshold).astype(int)

#     # 6. Отчет
#     print("\n=== Test Results ===")
#     try:
#         auc = roc_auc_score(yte, prob)
#         print(f"ROC AUC: {auc:.4f}")
#     except:
#         print("ROC AUC: N/A")
        
#     print(f"F1 Score: {f1_score(yte, pred):.4f}")
#     print(classification_report(yte, pred, digits=4))

# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument('--config', type=str, default='config.yaml')
#     ap.add_argument('--ckpt', type=str, required=True, help="Path to multimodal_clf.joblib")
#     ap.add_argument('--threshold', type=float, default=0.5)
#     args = ap.parse_args()

#     with open(args.config, 'r', encoding='utf-8') as f:
#         cfg = yaml.safe_load(f)
    
#     main(cfg, args.ckpt, args.threshold)


from __future__ import annotations
import sys
import os
sys.path.append(os.path.expanduser("~/work/python_libs"))

import argparse, yaml, joblib
import numpy as np
import torch
from sklearn.metrics import classification_report, roc_auc_score, f1_score

from src.datasets import load_jsonl, texts_images_labels
from src.encoders import TextEncoder, ImageEncoder
from src.utils import get_device

def fuse_features(text_emb, img_emb, texts, imgs, use_similarity: bool = True):
    """
    ВАЖНО: Должна полностью совпадать с логикой из train_multi.py (без Dropout)
    """
    Ht = text_emb.shape[1] if text_emb is not None else 0
    Hi = img_emb.shape[1]  if img_emb  is not None else 0

    feats = []
    img_positions = {idx: k for k, idx in enumerate([i for i, p in enumerate(imgs) if p])}

    for i in range(len(texts)):
        # Используем .copy(), чтобы не менять исходные эмбеддинги
        tvec = text_emb[i].copy() if text_emb is not None else np.zeros(Ht, dtype=np.float32)
        ivec = img_emb[img_positions[i]].copy() if (i in img_positions and img_emb is not None) else np.zeros(Hi, dtype=np.float32)

        parts = [tvec, ivec]

        sim = 0.0
        if use_similarity and Ht > 0 and Hi > 0:
            # Честная нормализация для косинусного сходства
            tv_norm = np.linalg.norm(tvec)
            iv_norm = np.linalg.norm(ivec)
            if tv_norm > 1e-9 and iv_norm > 1e-9:
                sim = float(np.dot(tvec / tv_norm, ivec / iv_norm))
        
        parts.append(np.array([sim], dtype=np.float32))
        feats.append(np.concatenate(parts, axis=0))

    return np.vstack(feats)

def main(cfg, ckpt, threshold: float):
    # 1. Загрузка бандла
    print(f"Loading model from {ckpt}...")
    bundle = joblib.load(ckpt)
    
    # Теперь достаем конфиг, который мы сохранили целиком
    saved_cfg = bundle.get('config', cfg) 
    clf = bundle['pipeline']

    # 2. Загрузка тестовых данных
    test_data = load_jsonl(cfg['paths']['test'])
    te_texts, te_imgs, yte = texts_images_labels(test_data)

    # Принудительно ставим ту же GPU, что при обучении
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # 3. Инициализация энкодеров по конфигу из бандла
    txt_enc = TextEncoder(
        saved_cfg["encoder"]["name"], 
        device, 
        max_length=saved_cfg["encoder"]["max_length"]
    )
    img_enc = ImageEncoder(saved_cfg["image_encoder"]["name"], device)

    # 4. Кодирование
    print("Encoding test data...")
    Xte_txt = txt_enc.encode(te_texts, batch_size=cfg['encoder']['batch_size'])
    
    te_img_idx = [i for i, p in enumerate(te_imgs) if p]
    Xte_img = img_enc.encode_paths(
        [te_imgs[i] for i in te_img_idx], 
        batch_size=cfg['image_encoder']['batch_size']
    ) if te_img_idx else None

    # 5. Слияние (Fusion)
    use_sim = bundle.get('multimodal', {}).get('use_similarity', True)
    Xte_fused = fuse_features(Xte_txt, Xte_img, te_texts, te_imgs, use_similarity=use_sim)

    # 6. Классификация
    print("Predicting...")
    prob = clf.predict_proba(Xte_fused)[:, 1]
    pred = (prob >= threshold).astype(int)

    # 7. Отчет
    print("\n" + "="*30)
    print("      TESTSET RESULTS")
    print("="*30)
    try:
        auc = roc_auc_score(yte, prob)
        print(f"ROC AUC:  {auc:.4f}")
    except Exception as e:
        print(f"ROC AUC:  N/A ({e})")
        
    print(f"F1 Score: {f1_score(yte, pred):.4f}")
    print("-" * 30)
    print(classification_report(yte, pred, digits=4))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='config.yaml')
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--threshold', type=float, default=0.5)
    args = ap.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    main(cfg, args.ckpt, args.threshold)