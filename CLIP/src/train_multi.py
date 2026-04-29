# from __future__ import annotations

# import sys
# import os
# sys.path.append(os.path.expanduser("~/work/python_libs"))


# import os, argparse, yaml, joblib
# import numpy as np
# from rich import print
# from sklearn.metrics import classification_report, roc_auc_score, f1_score
# import torch
# from src.datasets import load_jsonl, texts_images_labels
# from src.encoders import TextEncoder, ImageEncoder
# from src.models import make_clf
# from src.utils import set_seed, get_device, ensure_dir


# # def fuse(text_emb, img_emb, texts, imgs, use_similarity: bool = True):
# #     """
# #     Режим Image-Only: возвращаем только визуальные признаки.
# #     """
# #     # Сопоставление индексов для img_emb
# #     img_positions = {idx: k for k, idx in enumerate([i for i, p in enumerate(imgs) if p])}

# #     rows = []
# #     for i in range(len(texts)):
# #         # Берем ТОЛЬКО ivec (вектор картинки)
# #         if i in img_positions and img_emb is not None:
# #             ivec = img_emb[img_positions[i]]
# #         else:
# #             # Если картинки нет, создаем нулевой вектор (размер 512 для CLIP)
# #             ivec = np.zeros(512, dtype=np.float32)
        
# #         rows.append(ivec)
    
# #     return np.vstack(rows)

# def fuse(text_emb, img_emb, texts, imgs, use_similarity: bool = True):
#     """
#     Ранняя фузия: concat([text_emb], [image_emb], [similarity])
#     """
#     Ht = text_emb.shape[1] if text_emb is not None else 0
#     Hi = img_emb.shape[1]  if img_emb  is not None else 0

#     feats = []
#     # Сопоставление индексов для img_emb к исходным позициям
#     img_positions = {idx: k for k, idx in enumerate([i for i, p in enumerate(imgs) if p])}

#     for i in range(len(texts)):
#         tvec = text_emb[i] if text_emb is not None else None
#         ivec = img_emb[img_positions[i]] if (imgs[i] and img_emb is not None) else None

#         if tvec is None and ivec is None:
#             continue

#         if tvec is None:
#             tvec = np.zeros(Ht, dtype=np.float32)
#         if ivec is None:
#             ivec = np.zeros(Hi, dtype=np.float32)

#         parts = [tvec, ivec]

#         sim = 0.0
#         if use_similarity and Ht > 0 and Hi > 0 and (texts[i] and imgs[i]):
#             tv = tvec / (np.linalg.norm(tvec) + 1e-9)
#             iv = ivec / (np.linalg.norm(ivec) + 1e-9)
#             sim = float(np.dot(tv, iv))
#         parts.append(np.array([sim], dtype=np.float32))

#         feats.append(np.concatenate(parts, axis=0))

#     return np.vstack(feats)


# def filter_labels(texts, imgs, labels):
#     """Оставляем метки только для тех примеров, которые попали в фузию"""
#     keep = []
#     for i in range(len(labels)):
#         has_text = isinstance(texts[i], str) and len(texts[i]) > 0
#         has_img  = imgs[i] is not None
#         if has_text or has_img:
#             keep.append(i)
#     return np.array([labels[i] for i in keep])


# def main(cfg):
#     print("[magenta]=== Train multimodal (early fusion) ===[/magenta]")
#     set_seed(cfg.get("random_seed", 42))

#     # 1. Загрузка данных
#     train = load_jsonl(cfg["paths"]["train"])
#     val   = load_jsonl(cfg["paths"]["val"])
#     tr_texts, tr_imgs, ytr = texts_images_labels(train)
#     va_texts, va_imgs, yva = texts_images_labels(val)

#     # 2. Инициализация энкодеров
#     # dev_txt = get_device(cfg["encoder"].get("device", "auto"))
#     # dev_img = get_device(cfg["image_encoder"].get("device", "auto"))
#     gpu_idx =1 
#     dev_txt = torch.device(f"cuda:{gpu_idx}") if torch.cuda.is_available() else torch.device("cpu")
#     dev_img = torch.device(f"cuda:{gpu_idx}") if torch.cuda.is_available() else torch.device("cpu")

#     def get_gpu_name(d):
#         return torch.cuda.get_device_name(d) if 'cuda' in str(d) else 'CPU'
    
#     print(f"[bold green]>>> Устройство ТЕКСТ:[/bold green] {dev_txt} ({get_gpu_name(dev_txt)})")
#     print(f"[bold green]>>> Устройство ИЗОБР:[/bold green] {dev_img} ({get_gpu_name(dev_img)})")
    
#     txt_enc = TextEncoder(
#         cfg["encoder"]["name"],
#         dev_txt,
#         max_length=cfg["encoder"]["max_length"],
#         pool=cfg["features"].get("pool", "mean")
#     )
#     img_enc = ImageEncoder(cfg["image_encoder"]["name"], dev_img)

#     # 3. Кодирование текста
#     any_tr_text = any(isinstance(t, str) and len(t) > 0 for t in tr_texts)
#     any_va_text = any(isinstance(t, str) and len(t) > 0 for t in va_texts)
    
#     Xtr_txt = txt_enc.encode([t if isinstance(t, str) else "" for t in tr_texts],
#                              batch_size=cfg["encoder"]["batch_size"]) if any_tr_text else None
#     Xva_txt = txt_enc.encode([t if isinstance(t, str) else "" for t in va_texts],
#                              batch_size=cfg["encoder"]["batch_size"]) if any_va_text else None

#     # 4. Кодирование картинок
#     tr_img_idx = [i for i, p in enumerate(tr_imgs) if p]
#     va_img_idx = [i for i, p in enumerate(va_imgs) if p]

#     Xtr_img = img_enc.encode_paths([tr_imgs[i] for i in tr_img_idx],
#                                    batch_size=cfg["image_encoder"]["batch_size"]) if tr_img_idx else None
#     Xva_img = img_enc.encode_paths([va_imgs[i] for i in va_img_idx],
#                                    batch_size=cfg["image_encoder"]["batch_size"]) if va_img_idx else None

#     # 5. Фузия признаков
#     use_sim = cfg.get("multimodal", {}).get("use_similarity", True)
#     Xtr_fused = fuse(Xtr_txt, Xtr_img, tr_texts, tr_imgs, use_similarity=use_sim)
#     Xva_fused = fuse(Xva_txt, Xva_img, va_texts, va_imgs, use_similarity=use_sim)

#     ytr_fused = filter_labels(tr_texts, tr_imgs, ytr)
#     yva_fused = filter_labels(va_texts, va_imgs, yva)

#     print(f"Fused shapes: train={Xtr_fused.shape}, val={Xva_fused.shape}")

#     # 6. Обучение классификатора с циклом эпох
#     t_cfg = cfg.get("train", {})
#     max_epochs = int(t_cfg.get("epochs", 20))
#     patience = int(t_cfg.get("patience", 3))
    
#     # Создаем модель с поддержкой warm_start
#     clf = make_clf(
#         kind=t_cfg.get("clf_type", "logistic"),
#         normalize=cfg["features"]["normalize"],
#         lr=float(t_cfg.get("lr", 1e-3)),
#         weight_decay=float(t_cfg.get("weight_decay", 0.0)),
#         epochs=1,           # Один шаг за итерацию fit()
#         warm_start=True     # Дообучение вместо сброса
#     )

#     best_acc = -1.0
#     no_improve_count = 0

#     print(f"[green]Starting training: {t_cfg.get('clf_type')} for {max_epochs} epochs...[/green]")

#     for epoch in range(1, max_epochs + 1):
#         clf.fit(Xtr_fused, ytr_fused)
        
#         # Оценка текущей эпохи
#         tr_acc = clf.score(Xtr_fused, ytr_fused)
#         va_acc = clf.score(Xva_fused, yva_fused)
        
#         print(f"Epoch {epoch:02d}/{max_epochs} | Train Acc: {tr_acc:.4f} | Val Acc: {va_acc:.4f}")

#         # Проверка Early Stopping (терпение)
#         if va_acc > best_acc:
#             best_acc = va_acc
#             no_improve_count = 0
#         else:
#             no_improve_count += 1

#         if no_improve_count >= patience:
#             print(f"[yellow]Early stopping triggered! No improvement for {patience} epochs.[/yellow]")
#             break

#     # 7. Финальные метрики
#     prob = clf.predict_proba(Xva_fused)[:, 1]
#     pred = (prob >= 0.5).astype(int)
    
#     print("\n[bold]Final Results (Validation):[/bold]")
#     try:
#         print(f"ROC AUC: {roc_auc_score(yva_fused, prob):.4f}")
#     except:
#         print("ROC AUC: N/A (need more samples)")
    
#     print(classification_report(yva_fused, pred, digits=4))

#     # 8. Сохранение бандла
#     ensure_dir(cfg["paths"]["outdir"])
#     bundle = {
#         "pipeline": clf,
#         "text": {
#             "encoder_name": cfg["encoder"]["name"],
#             "max_length": cfg["encoder"]["max_length"],
#             "pool": cfg["features"].get("pool", "mean"),
#             "dim": int(Xtr_txt.shape[1]) if Xtr_txt is not None else 0,
#         },
#         "image": {
#             "encoder_name": cfg["image_encoder"]["name"],
#             "dim": int(Xtr_img.shape[1]) if Xtr_img is not None else 0,
#         },
#         "multimodal": cfg.get("multimodal", {})
#     }
 
#     outpath = os.path.join(cfg["paths"]["outdir"], "multimodal_clf.joblib")
#     joblib.dump(bundle, outpath)
#     print(f"[magenta]Model saved to ->[/magenta] {outpath}")


# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--config", type=str, default="config.yaml")
#     args = ap.parse_args()
    
#     with open(args.config, "r", encoding="utf-8") as f:
#         config = yaml.safe_load(f)
#     main(config)



from __future__ import annotations
import sys
import os
import random

# Добавь свои пути, если библиотеки лежат в нестандартных папках
sys.path.append(os.path.expanduser("~/work/python_libs"))

import argparse, yaml, joblib
import numpy as np
from rich import print
from sklearn.metrics import classification_report, roc_auc_score
import torch

from src.datasets import load_jsonl, texts_images_labels
from src.encoders import TextEncoder, ImageEncoder
from src.models import make_clf
from src.utils import set_seed, ensure_dir

def fuse(text_emb, img_emb, texts, imgs, use_similarity: bool = True, dropout_prob: float = 0.0):
    """
    Ранняя фузия признаков.
    dropout_prob: вероятность зануления текста (используется только при обучении).
    """
    Ht = text_emb.shape[1] if text_emb is not None else 0
    Hi = img_emb.shape[1] if img_emb is not None else 0

    feats = []
    # Сопоставление индексов для img_emb к исходным позициям в списке текстов
    img_positions = {idx: k for k, idx in enumerate([i for i, p in enumerate(imgs) if p])}

    for i in range(len(texts)):
        tvec = text_emb[i].copy() if text_emb is not None else None
        ivec = img_emb[img_positions[i]].copy() if (i in img_positions and img_emb is not None) else None

        # Пропускаем, если нет данных совсем
        if tvec is None and ivec is None:
            continue

        # Заглушки для отсутствующих модальностей
        if tvec is None: tvec = np.zeros(Ht, dtype=np.float32)
        if ivec is None: ivec = np.zeros(Hi, dtype=np.float32)

        # --- TEXT DROPOUT (Критично для ВКР) ---
        # С заданной вероятностью зануляем текст, чтобы модель училась на картинке
        if dropout_prob > 0 and random.random() < dropout_prob:
            tvec = np.zeros_like(tvec)

        parts = [tvec, ivec]

        # Добавляем Similarity (Косинусное сходство)
        sim = 0.0
        if use_similarity and Ht > 0 and Hi > 0:
            # Нормализация для честного косинуса
            tv_norm = np.linalg.norm(tvec)
            iv_norm = np.linalg.norm(ivec)
            if tv_norm > 1e-9 and iv_norm > 1e-9:
                sim = float(np.dot(tvec / tv_norm, ivec / iv_norm))
        
        parts.append(np.array([sim], dtype=np.float32))
        feats.append(np.concatenate(parts, axis=0))

    return np.vstack(feats)

def filter_labels(texts, imgs, labels):
    """Оставляем метки только для тех примеров, где есть хоть одна модальность"""
    keep = []
    for i in range(len(labels)):
        has_text = isinstance(texts[i], str) and len(texts[i]) > 0
        has_img  = imgs[i] is not None
        if has_text or has_img:
            keep.append(i)
    return np.array([labels[i] for i in keep])

def main(cfg):
    print("[magenta]=== Train multimodal (early fusion) ===[/magenta]")
    set_seed(cfg.get("random_seed", 42))

    # 1. Загрузка данных
    train = load_jsonl(cfg["paths"]["train"])
    val   = load_jsonl(cfg["paths"]["val"])
    tr_texts, tr_imgs, ytr = texts_images_labels(train)
    va_texts, va_imgs, yva = texts_images_labels(val)

    # 2. Настройка GPU
    gpu_idx = 1 
    device = torch.device(f"cuda:{gpu_idx}" if torch.cuda.is_available() else "cpu")
    print(f"[bold green]>>> Использование устройства:[/bold green] {device}")

    # 3. Инициализация энкодеров
    txt_enc = TextEncoder(cfg["encoder"]["name"], device, 
                          max_length=cfg["encoder"]["max_length"])
    img_enc = ImageEncoder(cfg["image_encoder"]["name"], device)

    # 4. Кодирование (Cache embeddings)
    print("Encoding texts and images...")
    Xtr_txt = txt_enc.encode(tr_texts, batch_size=cfg["encoder"]["batch_size"])
    Xva_txt = txt_enc.encode(va_texts, batch_size=cfg["encoder"]["batch_size"])

    tr_img_idx = [i for i, p in enumerate(tr_imgs) if p]
    va_img_idx = [i for i, p in enumerate(va_imgs) if p]
    
    Xtr_img = img_enc.encode_paths([tr_imgs[i] for i in tr_img_idx], batch_size=cfg["image_encoder"]["batch_size"])
    Xva_img = img_enc.encode_paths([va_imgs[i] for i in va_img_idx], batch_size=cfg["image_encoder"]["batch_size"])

    # 5. Подготовка классификатора
    t_cfg = cfg.get("train", {})
    # Важно: преобразуем lr к float, чтобы избежать ошибок YAML
    learning_rate = float(t_cfg.get("lr", 0.001))
    
    clf = make_clf(
        kind=t_cfg.get("clf_type", "mlp"),
        normalize=cfg["features"].get("normalize", True),
        lr=learning_rate,
        epochs=1, 
        warm_start=True # Позволяет дообучать модель в цикле
    )

    # 6. Цикл обучения
    max_epochs = int(t_cfg.get("epochs", 20))
    patience = int(t_cfg.get("patience", 3))
    use_sim = cfg.get("multimodal", {}).get("use_similarity", True)
    
    best_acc = -1.0
    no_improve = 0

    print(f"[yellow]Starting training for {max_epochs} epochs...[/yellow]")
    
    for epoch in range(1, max_epochs + 1):
        # При обучении используем Dropout текста (30%), чтобы заставить модель смотреть на картинки
        Xtr_fused = fuse(Xtr_txt, Xtr_img, tr_texts, tr_imgs, use_similarity=use_sim, dropout_prob=0.3)
        ytr_fused = filter_labels(tr_texts, tr_imgs, ytr)
        
        clf.fit(Xtr_fused, ytr_fused)
        
        # На валидации Dropout НЕ используем (dropout_prob=0)
        Xva_fused = fuse(Xva_txt, Xva_img, va_texts, va_imgs, use_similarity=use_sim, dropout_prob=0.0)
        yva_fused = filter_labels(va_texts, va_imgs, yva)
        
        va_acc = clf.score(Xva_fused, yva_fused)
        print(f"Epoch {epoch:02d} | Val Acc: {va_acc:.4f}")

        if va_acc > best_acc:
            best_acc = va_acc
            no_improve = 0
            # Сохраняем "лучшее" состояние можно здесь
        else:
            no_improve += 1
        
        if no_improve >= patience:
            print("[red]Early stopping![/red]")
            break

    # 7. Отчет
    final_probs = clf.predict_proba(Xva_fused)[:, 1]
    final_preds = (final_probs >= 0.5).astype(int)
    print("\n[bold]Final Report:[/bold]")
    print(classification_report(yva_fused, final_preds, digits=4))

    # 8. Сохранение
    ensure_dir(cfg["paths"]["outdir"])
    bundle = {
        "pipeline": clf,
        "config": cfg,
        "multimodal": cfg.get("multimodal", {})
    }
    joblib.dump(bundle, os.path.join(cfg["paths"]["outdir"], "multimodal_clf.joblib"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    args = ap.parse_args()
    with open(args.config, "r") as f:
        main(yaml.safe_load(f))