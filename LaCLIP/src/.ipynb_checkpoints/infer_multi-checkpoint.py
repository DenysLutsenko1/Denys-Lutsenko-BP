from __future__ import annotations

import sys
import os
sys.path.append(os.path.expanduser("~/work/python_libs"))

import argparse, joblib, yaml, os, numpy as np
from src.encoders import TextEncoder, ImageEncoder
from src.utils import get_device


def infer(bundle_path: str, texts: list[str|None], image_paths: list[str|None]):
    bundle = joblib.load(bundle_path)
    mm_cfg = bundle.get("multimodal", {})
    use_sim = mm_cfg.get("use_similarity", True)

    # ЭНКОДЕРЫ
    dev_txt = get_device("auto")
    dev_img = get_device("auto")
    txt_enc = TextEncoder(
        bundle["text"]["encoder_name"],
        dev_txt,
        max_length=bundle["text"]["max_length"],
        pool=bundle["text"]["pool"]
    )
    img_enc = ImageEncoder(bundle["image"]["encoder_name"], dev_img)

    # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Берем размерности из сохраненного бандла
    # Если в бандле нет ключа 'dim', для CLIP это всегда 512
    Ht = bundle["text"].get("dim", 512)
    Hi = bundle["image"].get("dim", 512)

    # ОБРАБОТКА ТЕКСТА
    txts = [t if isinstance(t, str) else "" for t in texts]
    X_txt = txt_enc.encode(txts, batch_size=64) if len(txts) else None

    # ОБРАБОТКА ИЗОБРАЖЕНИЙ
    # Фильтруем только существующие пути
    idx_img = [i for i, p in enumerate(image_paths) if p and os.path.exists(p)]
    if idx_img:
        X_img_raw = img_enc.encode_paths([image_paths[i] for i in idx_img], batch_size=32)
    else:
        X_img_raw = None
    
    # Словарь для быстрого доступа к вектору картинки по индексу строки
    pos = {i: k for k, i in enumerate(idx_img)}

    rows = []
    # Определяем количество строк для обработки
    n_rows = max(len(texts), len(image_paths))
    
    for i in range(n_rows):
        # 1. Текстовый вектор
        if X_txt is not None and i < len(X_txt):
            tvec = X_txt[i]
        else:
            tvec = np.zeros(Ht, dtype=np.float32)

        # 2. Визуальный вектор
        if i in pos and X_img_raw is not None:
            ivec = X_img_raw[pos[i]]
        else:
            ivec = np.zeros(Hi, dtype=np.float32)

        # 3. Similarity (Схожесть)
        sim = 0.0
        text_present = i < len(texts) and isinstance(texts[i], str) and len(texts[i]) > 0
        image_present = i < len(image_paths) and image_paths[i] is not None
        
        if use_sim and text_present and image_present:
            # Считаем косинусное сходство, если есть оба компонента
            tv_norm = np.linalg.norm(tvec)
            iv_norm = np.linalg.norm(ivec)
            if tv_norm > 1e-9 and iv_norm > 1e-9:
                tv = tvec / tv_norm
                iv = ivec / iv_norm
                sim = float(np.dot(tv, iv))
        
        # Склеиваем: 512 (текст) + 512 (фото) + 1 (sim) = 1025 признаков
        row = np.concatenate([tvec, ivec, np.array([sim], dtype=np.float32)], axis=0)
        rows.append(row)

    X = np.vstack(rows)

    # ПРЕДСКАЗАНИЕ
    clf = bundle["pipeline"]
    prob = clf.predict_proba(X)[:, 1]
    return prob


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