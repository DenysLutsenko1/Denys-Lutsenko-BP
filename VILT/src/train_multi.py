# from __future__ import annotations

# import sys
# import os
# sys.path.append(os.path.expanduser("~/work/python_libs"))


# import os, yaml, joblib
# os.environ["HF_HUB_DISABLE_TORCH_SAFE_LOAD_CHECK"] = "1"
# import numpy as np
# import torch
# from rich import print
# from src.datasets import load_jsonl, texts_images_labels
# from src.encoders import ViltEncoder
# from src.models import make_clf
# from src.utils import set_seed, get_device, ensure_dir

# def main(cfg):
#     set_seed(cfg.get("random_seed", 42))
    
#     gpu_index = 1  # Просто меняй эту цифру на нужную (0, 1, 2...)
#     device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")

    
#     # Извлекаем max_length из конфига (по умолчанию 40, если забыли указать)
#     max_len = cfg["vilt"].get("max_length", 40)

#     print(f"[bold green]>>> Текущее устройство:[/bold green] {device}")
#     print(f"[bold blue]>>> Лимит токенов текста:[/bold blue] {max_len}")

#     # 1. Загрузка данных
#     train = load_jsonl(cfg["paths"]["train"])
#     val = load_jsonl(cfg["paths"]["val"])
#     tr_txt, tr_img, ytr = texts_images_labels(train)
#     va_txt, va_img, yva = texts_images_labels(val)

#     # 2. Кодирование ViLT 
#     # Передаем max_len в инициализацию энкодера
#     encoder = ViltEncoder(cfg["vilt"]["name"], device, max_length=max_len)
    
#     print(f"[yellow]Encoding train data with ViLT (max_len={max_len})...[/yellow]")
#     Xtr = encoder.encode(tr_txt, tr_img, batch_size=cfg["vilt"]["batch_size"])
    
#     print("[yellow]Encoding val data...[/yellow]")
#     Xva = encoder.encode(va_txt, va_img, batch_size=cfg["vilt"]["batch_size"])

#     # 3. Обучение классификатора
#     clf = make_clf(
#         kind=cfg["train"]["clf_type"],
#         epochs=1,
#         warm_start=True
#     )

#     best_acc = 0
#     patience = cfg["train"]["patience"]
#     wait = 0

#     for epoch in range(1, cfg["train"]["epochs"] + 1):
#         clf.fit(Xtr, ytr)
#         acc = clf.score(Xva, yva)
#         print(f"Epoch {epoch} | Val Acc: {acc:.4f}")
        
#         if acc > best_acc:
#             best_acc = acc
#             wait = 0
#         else:
#             wait += 1
#         if wait >= patience: 
#             print(f"[red]Early stopping at epoch {epoch}[/red]")
#             break

#     # 4. Сохранение
#     ensure_dir(cfg["paths"]["outdir"])
    
#     # ВАЖНО: сохраняем max_length в бандл, чтобы app.py его увидел
#     bundle = {
#         "pipeline": clf, 
#         "model_name": cfg["vilt"]["name"],
#         "max_length": max_len 
#     }
    
#     out_path = os.path.join(cfg["paths"]["outdir"], "multimodal_clf.joblib")
#     joblib.dump(bundle, out_path)
#     print(f"[bold green]Успешно сохранено в: {out_path}[/bold green]")

# if __name__ == "__main__":
#     with open("config.yaml", "r") as f:
#         main(yaml.safe_load(f))

from __future__ import annotations
import sys
import os
sys.path.append(os.path.expanduser("~/work/python_libs"))

import os, yaml, joblib, torch
import numpy as np
from rich import print
from src.datasets import load_jsonl, texts_images_labels
from src.encoders import ViltEncoder
from src.models import make_clf
from src.utils import set_seed, ensure_dir

def main(cfg):
    set_seed(cfg.get("random_seed", 42))
    
    # Настройка устройства
    gpu_index = 1 
    device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")
    max_len = cfg["vilt"].get("max_length", 40)

    print(f"[bold green]>>> Устройство:[/bold green] {device} | [bold blue]Max Len:[/bold blue] {max_len}")

    # 1. Загрузка данных
    train = load_jsonl(cfg["paths"]["train"])
    val = load_jsonl(cfg["paths"]["val"])
    tr_txt, tr_img, ytr = texts_images_labels(train)
    va_txt, va_img, yva = texts_images_labels(val)

    # 2. Инициализация энкодера
    encoder = ViltEncoder(cfg["vilt"]["name"], device, max_length=max_len)
    
    # 3. Кодирование с применением TEXT DROPOUT
    # dropout_prob=0.5 заставит ViLT в половине случаев игнорировать текст
    print(f"[yellow]Encoding TRAIN (Text Dropout 0.5 active)...[/yellow]")
    Xtr = encoder.encode(tr_txt, tr_img, batch_size=cfg["vilt"]["batch_size"], dropout_prob=0.5)
    
    # На валидации дропаут всегда 0, чтобы оценить реальную силу модели
    print("[yellow]Encoding VAL (No Dropout)...[/yellow]")
    Xva = encoder.encode(va_txt, va_img, batch_size=cfg["vilt"]["batch_size"], dropout_prob=0.0)

    # 4. Обучение классификатора (MLP)
    clf = make_clf(
        kind=cfg["train"]["clf_type"],
        epochs=1,
        warm_start=True
    )

    best_acc = 0
    patience = cfg["train"].get("patience", 5)
    wait = 0

    print("[bold magenta]Starting training epochs...[/bold magenta]")
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        clf.fit(Xtr, ytr)
        acc = clf.score(Xva, yva)
        print(f"Epoch {epoch:02d} | Val Acc: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            wait = 0
        else:
            wait += 1
            
        if wait >= patience: 
            print(f"[red]Early stopping![/red]")
            break

    # 5. Сохранение бандла
    ensure_dir(cfg["paths"]["outdir"])
    bundle = {
        "pipeline": clf, 
        "model_name": cfg["vilt"]["name"],
        "max_length": max_len 
    }
    
    out_path = os.path.join(cfg["paths"]["outdir"], "multimodal_clf.joblib")
    joblib.dump(bundle, out_path)
    print(f"[bold green]Модель сохранена: {out_path}[/bold green]")

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        main(yaml.safe_load(f))