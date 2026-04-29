from __future__ import annotations

import sys
import os
sys.path.append(os.path.expanduser("~/work/python_libs"))

# только карттинки 
# import argparse, joblib, os, torch
# import numpy as np
# from src.encoders import ImageEncoder
# from src.utils import get_device

# def infer(bundle_path: str, image_paths: list[str]):
#     # 1. Загружаем бандл
#     bundle = joblib.load(bundle_path)
#     clf = bundle["pipeline"]
    
#     # 2. Инициализируем только ИЗОБРАЖЕНИЕ
#     device = get_device("auto")
#     img_enc = ImageEncoder(bundle["image"]["encoder_name"], device)

#     # 3. Проверяем существование файлов
#     valid_paths = []
#     for p in image_paths:
#         if p and os.path.exists(p):
#             valid_paths.append(p)
#         else:
#             print(f"[Warning] Файл не найден: {p}")

#     if not valid_paths:
#         print("[Error] Нет доступных изображений для обработки!")
#         return np.array([])

#     # 4. Кодируем изображения (получаем те самые 512 признаков)
#     print(f"Encoding {len(valid_paths)} images...")
#     X_img = img_enc.encode_paths(valid_paths, batch_size=32)

#     # 5. Предсказание (модель теперь ждет именно 512 признаков)
#     probs = clf.predict_proba(X_img)[:, 1]
#     return probs

# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--ckpt", type=str, required=True)
#     ap.add_argument("--image", type=str, nargs="+", required=True)
#     # Аргумент --text оставляем для совместимости, но не используем
#     ap.add_argument("--text", type=str, nargs="*", default=[]) 
#     args = ap.parse_args()

#     probs = infer(args.ckpt, args.image)
    
#     if len(probs) > 0:
#         for i, p in enumerate(probs):
#             print(f"{i:02d} p(genai)={p:.3f} image='{args.image[i]}'")


import argparse, joblib, yaml, os, numpy as np
from src.encoders import TextEncoder, ImageEncoder
from src.utils import get_device
import torch
import numpy as np

# старое без дропаут
# def infer(bundle_path: str, texts: list[str|None], image_paths: list[str|None]):
#     bundle = joblib.load(bundle_path)
#     mm_cfg = bundle.get("multimodal", {})
#     use_sim = mm_cfg.get("use_similarity", True)

#     # ЭНКОДЕРЫ
#     dev_txt = get_device("auto")
#     dev_img = get_device("auto")
#     txt_enc = TextEncoder(
#         bundle["text"]["encoder_name"],
#         dev_txt,
#         max_length=bundle["text"]["max_length"],
#         pool=bundle["text"]["pool"]
#     )
#     img_enc = ImageEncoder(bundle["image"]["encoder_name"], dev_img)

#     # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Берем размерности из сохраненного бандла
#     # Если в бандле нет ключа 'dim', для CLIP это всегда 512
#     Ht = bundle["text"].get("dim", 512)
#     Hi = bundle["image"].get("dim", 512)

#     # ОБРАБОТКА ТЕКСТА
#     txts = [t if isinstance(t, str) else "" for t in texts]
#     X_txt = txt_enc.encode(txts, batch_size=64) if len(txts) else None

#     # ОБРАБОТКА ИЗОБРАЖЕНИЙ
#     # Фильтруем только существующие пути
#     idx_img = [i for i, p in enumerate(image_paths) if p and os.path.exists(p)]
#     if idx_img:
#         X_img_raw = img_enc.encode_paths([image_paths[i] for i in idx_img], batch_size=32)
#     else:
#         X_img_raw = None
    
#     # Словарь для быстрого доступа к вектору картинки по индексу строки
#     pos = {i: k for k, i in enumerate(idx_img)}

#     rows = []
#     # Определяем количество строк для обработки
#     n_rows = max(len(texts), len(image_paths))
    
#     for i in range(n_rows):
#         # 1. Текстовый вектор
#         if X_txt is not None and i < len(X_txt):
#             tvec = X_txt[i]
#         else:
#             tvec = np.zeros(Ht, dtype=np.float32)

#         # 2. Визуальный вектор
#         if i in pos and X_img_raw is not None:
#             ivec = X_img_raw[pos[i]]
#         else:
#             ivec = np.zeros(Hi, dtype=np.float32)

#         # 3. Similarity (Схожесть)
#         sim = 0.0
#         text_present = i < len(texts) and isinstance(texts[i], str) and len(texts[i]) > 0
#         image_present = i < len(image_paths) and image_paths[i] is not None
        
#         if use_sim and text_present and image_present:
#             # Считаем косинусное сходство, если есть оба компонента
#             tv_norm = np.linalg.norm(tvec)
#             iv_norm = np.linalg.norm(ivec)
#             if tv_norm > 1e-9 and iv_norm > 1e-9:
#                 tv = tvec / tv_norm
#                 iv = ivec / iv_norm
#                 sim = float(np.dot(tv, iv))
        
#         # Склеиваем: 512 (текст) + 512 (фото) + 1 (sim) = 1025 признаков
#         row = np.concatenate([tvec, ivec, np.array([sim], dtype=np.float32)], axis=0)
#         rows.append(row)

#     X = np.vstack(rows)

#     # ПРЕДСКАЗАНИЕ
#     clf = bundle["pipeline"]
#     prob = clf.predict_proba(X)[:, 1]
#     return prob



def fuse(text_emb, img_emb, texts, imgs, use_similarity: bool = True, dropout_prob: float = 0.0):
    """
    Функция фузии для инференса (dropout_prob всегда 0.0 здесь).
    """
    Ht = text_emb.shape[1] if text_emb is not None else 0
    Hi = img_emb.shape[1] if img_emb is not None else 0

    feats = []
    img_positions = {idx: k for k, idx in enumerate([i for i, p in enumerate(imgs) if p])}

    for i in range(len(texts)):
        tvec = text_emb[i].copy() if text_emb is not None else None
        ivec = img_emb[img_positions[i]].copy() if (i in img_positions and img_emb is not None) else None

        if tvec is None: tvec = np.zeros(Ht, dtype=np.float32)
        if ivec is None: ivec = np.zeros(Hi, dtype=np.float32)

        # Для инференса мы никогда не зануляем текст случайно (dropout_prob=0)
        
        parts = [tvec, ivec]

        sim = 0.0
        if use_similarity and Ht > 0 and Hi > 0:
            tv_norm = np.linalg.norm(tvec)
            iv_norm = np.linalg.norm(ivec)
            if tv_norm > 1e-9 and iv_norm > 1e-9:
                sim = float(np.dot(tvec / tv_norm, ivec / iv_norm))
        
        parts.append(np.array([sim], dtype=np.float32))
        feats.append(np.concatenate(parts, axis=0))

    return np.vstack(feats)

def infer(ckpt_path, texts, image_paths):
    # Загружаем бандл
    bundle = joblib.load(ckpt_path)
    clf = bundle["pipeline"]
    
    # Достаем конфиг из бандла (теперь он лежит там)
    cfg = bundle.get("config")
    
    # Определяем устройство
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    # Инициализируем энкодеры, используя данные из сохраненного конфига
    txt_enc = TextEncoder(
        cfg["encoder"]["name"], 
        device, 
        max_length=cfg["encoder"]["max_length"]
    )
    img_enc = ImageEncoder(cfg["image_encoder"]["name"], device)

    # Кодируем
    X_txt = txt_enc.encode(texts, batch_size=1)
    X_img = img_enc.encode_paths(image_paths, batch_size=1)

    # Склеиваем признаки (важно использовать ту же логику fuse, что и при обучении!)
    # Но для инференса dropout всегда 0
    use_sim = cfg.get("multimodal", {}).get("use_similarity", True)
    
    # Вызываем ту же функцию fuse, которую мы правили в train_multi
    # Убедись, что она импортирована или скопирована сюда
    X_fused = fuse(X_txt, X_img, texts, image_paths, use_similarity=use_sim, dropout_prob=0.0)

    probs = clf.predict_proba(X_fused)[:, 1]
    return probs



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--text", type=str, nargs="*", default=[])
    ap.add_argument("--image", type=str, nargs="*", default=[])
    args = ap.parse_args()

    # Выравниваем списки
    n = max(len(args.text), len(args.image), 1)
    # Если передали только текст, список картинок будет состоять из None
    texts = args.text + [None] * (n - len(args.text))
    images = args.image + [None] * (n - len(args.image))

    probs = infer(args.ckpt, texts, images)
    for i, p in enumerate(probs):
        print(f"{i:02d} p(genai)={p:.3f}  text={texts[i]!r}  image={images[i]!r}")