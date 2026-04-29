from __future__ import annotations

import sys
import os
sys.path.append(os.path.expanduser("~/work/python_libs"))

import os, argparse, yaml, joblib
import numpy as np
from rich import print
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import torch
from src.datasets import load_jsonl, texts_aug_images_labels
from src.encoders import TextEncoder, ImageEncoder
from src.models import make_clf
from src.utils import set_seed, get_device, ensure_dir

def fuse(text_emb, img_emb, texts, imgs, use_similarity: bool = True):
    """
    Ранняя фузия: concat([text_emb], [image_emb], [similarity])
    """
    Ht = text_emb.shape[1] if text_emb is not None else 0
    Hi = img_emb.shape[1]  if img_emb  is not None else 0

    feats = []
    # Индексы для картинок (так как не у всех постов есть фото)
    img_positions = {idx: k for k, idx in enumerate([i for i, p in enumerate(imgs) if p])}

    for i in range(len(texts)):
        tvec = text_emb[i] if text_emb is not None else np.zeros(Ht, dtype=np.float32)
        
        if i in img_positions:
            ivec = img_emb[img_positions[i]]
        else:
            ivec = np.zeros(Hi, dtype=np.float32)

        # Считаем Similarity (косинусное расстояние) для LaCLIP признака
        sim = 0.0
        if use_similarity and (i in img_positions) and (texts[i] is not None):
            t_norm = np.linalg.norm(tvec)
            i_norm = np.linalg.norm(ivec)
            if t_norm > 1e-9 and i_norm > 1e-9:
                sim = float(np.dot(tvec / t_norm, ivec / i_norm))

        row = np.concatenate([tvec, ivec, np.array([sim], dtype=np.float32)], axis=0)
        feats.append(row)

    return np.vstack(feats)

def main(cfg):
    set_seed(cfg.get("seed", 42))
    
    # 1. Загрузка данных (используем новую функцию с поддержкой аугментаций)
    print("[yellow]Loading data...[/yellow]")
    train_data = load_jsonl(cfg["paths"]["train"])
    val_data = load_jsonl(cfg["paths"]["val"])
    
    tr_txt, tr_aug, tr_img_paths, ytr = texts_aug_images_labels(train_data)
    va_txt, va_aug, va_img_paths, yva = texts_aug_images_labels(val_data)

    dev_txt = get_device(cfg["encoder"].get("device", "auto"))
    dev_img = get_device(cfg["image_encoder"].get("device", "auto"))

    def get_gpu_name(d):
        return torch.cuda.get_device_name(0) if 'cuda' in str(d) else 'CPU'
    
    print(f"[bold green]>>> Устройство ТЕКСТ:[/bold green] {dev_txt} ({get_gpu_name(dev_txt)})")
    print(f"[bold green]>>> Устройство ИЗОБР:[/bold green] {dev_img} ({get_gpu_name(dev_img)})")

    # 2. Инициализация энкодеров
    txt_enc = TextEncoder(cfg["encoder"]["name"], dev_txt, max_length=cfg["encoder"]["max_length"])
    img_enc = ImageEncoder(cfg["image_encoder"]["name"], dev_img)

    # 3. Кодирование ТЕКСТА (LaCLIP: оригинал + аугментации)
    print("[cyan]Encoding texts with LaCLIP...[/cyan]")
    Xtr_txt = txt_enc.encode_laclip(tr_txt, tr_aug)
    Xva_txt = txt_enc.encode_laclip(va_txt, va_aug)

    # 4. Кодирование КАРТИНОК
    print("[cyan]Encoding images...[/cyan]")
    tr_imgs_to_proc = [p for p in tr_img_paths if p]
    va_imgs_to_proc = [p for p in va_img_paths if p]
    
    Xtr_img = img_enc.encode_paths(tr_imgs_to_proc, batch_size=cfg["image_encoder"]["batch_size"])
    Xva_img = img_enc.encode_paths(va_imgs_to_proc, batch_size=cfg["image_encoder"]["batch_size"])

    # 5. Фузия признаков
    print("[yellow]Fusing features...[/yellow]")
    use_sim = cfg.get("multimodal", {}).get("use_similarity", True)
    Xtr_fused = fuse(Xtr_txt, Xtr_img, tr_txt, tr_img_paths, use_similarity=use_sim)
    Xva_fused = fuse(Xva_txt, Xva_img, va_txt, va_img_paths, use_similarity=use_sim)

    # 6. Обучение классификатора
    # Подтягиваем всю секцию train из конфига
    t_cfg = cfg.get("train", {})
    
    # Извлекаем параметры (если в конфиге их нет, используем значения по умолчанию)
    kind = t_cfg.get("clf_type", "mlp")
    max_epochs = int(t_cfg.get("epochs", 20))    # <-- Вот здесь подтягиваются эпохи
    patience = int(t_cfg.get("patience", 5))     # <-- Подтягиваем терпение
    lr = float(t_cfg.get("lr", 1e-3))            # <-- Подтягиваем скорость обучения
    alpha_val = float(t_cfg.get("weight_decay", 0.0001))
    
    print(f"[green]Starting training: {kind} for {max_epochs} epochs (from config)...[/green]")
    
    # Создаем модель. Важно: epochs=1, так как мы сами крутим цикл max_epochs раз
    clf = make_clf(
        kind=kind,
        lr=lr,
        weight_decay=alpha_val,
        epochs=1,
        warm_start=True  # Чтобы модель училась постепенно, а не заново
    )

    best_acc = 0
    no_improve_count = 0

    # Цикл по количеству эпох из конфига
    for epoch in range(1, max_epochs + 1):
        clf.fit(Xtr_fused, ytr)
        
        # Считаем точность на валидации
        val_acc = clf.score(Xva_fused, yva)
        train_acc = clf.score(Xtr_fused, ytr)
        
        print(f"Epoch {epoch}/{max_epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        # Проверка Early Stopping (терпение)
        if val_acc > best_acc:
            best_acc = val_acc
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            print(f"[yellow]Early stopping! No improvement for {patience} epochs (from config).[/yellow]")
            break

    # 7. Валидация
    pred = clf.predict(Xva_fused)
    prob = clf.predict_proba(Xva_fused)[:, 1]
    
    print("\n[bold magenta]Results:[/bold magenta]")
    print(f"ROC AUC: {roc_auc_score(yva, prob):.4f}")
    print(classification_report(yva, pred, digits=4))

    # 8. Сохранение бандла
    ensure_dir(cfg["paths"]["outdir"])
    bundle = {
        "pipeline": clf,
        "text": {
            "encoder_name": cfg["encoder"]["name"],
            "max_length": cfg["encoder"]["max_length"],
            "pool": cfg.get("features", {}).get("pool", "mean"),
            "dim": int(Xtr_txt.shape[1])
        },
        "image": {
            "encoder_name": cfg["image_encoder"]["name"],
            "dim": int(Xtr_img.shape[1])
        },
        "multimodal": cfg.get("multimodal", {})
    }
    outpath = os.path.join(cfg["paths"]["outdir"], "multimodal.joblib")
    joblib.dump(bundle, outpath)
    print(f"[green]Model saved to {outpath}[/green]")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    args = ap.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    main(config)