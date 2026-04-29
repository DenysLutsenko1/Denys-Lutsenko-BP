# from __future__ import annotations
# import sys
# import os
# sys.path.append(os.path.expanduser("~/work/python_libs"))

# import yaml
# import joblib
# import argparse
# import numpy as np
# from rich import print

# from src.datasets import load_jsonl, texts_images_labels
# from src.encoders import Blip2Encoder 
# from src.models import make_clf
# from src.utils import set_seed, get_device, ensure_dir, log_vram, clear_gpu_cache


# def main(cfg):
#     set_seed(cfg.get("random_seed", 42))
#     device = get_device(cfg["blip2"].get("device", "auto"))

#     # 1. Загрузка данных
#     train = load_jsonl(cfg["paths"]["train"])
#     val   = load_jsonl(cfg["paths"]["val"])

#     tr_txt, tr_img, ytr = texts_images_labels(train)
#     va_txt, va_img, yva = texts_images_labels(val)

#     # 2. Инициализация моделей (BLIP-2 + SBERT)
#     encoder = Blip2Encoder(cfg["blip2"]["name"], device)
#     log_vram()

#     # 3. Кодирование в векторы
#     print("[yellow]Encoding train...[/yellow]")
#     Xtr = encoder.encode(tr_txt, tr_img, batch_size=cfg["blip2"]["batch_size"])

#     print("[yellow]Encoding val...[/yellow]")
#     Xva = encoder.encode(va_txt, va_img, batch_size=cfg["blip2"]["batch_size"])

#     clear_gpu_cache()

#     # Защита от NaN (критично для маленьких датасетов)
#     Xtr = np.nan_to_num(Xtr.astype(np.float32))
#     Xva = np.nan_to_num(Xva.astype(np.float32))
    
#     print(f"[cyan]Embeddings ready: train={Xtr.shape}, val={Xva.shape}[/cyan]")

#     # 4. Создание классификатора (без Scaler)
#     clf = make_clf(
#         kind=cfg["train"]["clf_type"],
#         normalize=False, 
#         lr=float(cfg["train"]["lr"]),
#         epochs=1,
#         warm_start=True
#     )

#     best_acc = -1.0
#     patience = cfg["train"]["patience"]
#     wait     = 0
#     ensure_dir(cfg["paths"]["outdir"])
#     save_path = os.path.join(cfg["paths"]["outdir"], "hybrid_model.joblib")

#     # 5. Цикл обучения
#     for epoch in range(1, cfg["train"]["epochs"] + 1):
#         clf.fit(Xtr, ytr) # Обучаем напрямую на Xtr
#         acc = clf.score(Xva, yva)
        
#         print(f"Epoch {epoch:3d} | Val Acc: {acc:.4f} | Best: {best_acc:.4f} | Wait: {wait}/{patience}")

#         if acc > best_acc:
#             best_acc = acc
#             wait = 0
#             joblib.dump({"pipeline": clf, "model_name": "blip2+sbert"}, save_path)
#             print(f"[green]New best model saved![/green]")
#         else:
#             wait += 1

#         if wait >= patience:
#             print(f"[red]Early stopping.[/red]")
#             break

#     print(f"[bold green]Testing finished! Best Val Acc: {best_acc:.4f}[/bold green]")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", type=str, default="config.yaml")
#     args = parser.parse_args()

#     with open(args.config, "r", encoding="utf-8") as f:
#         config_data = yaml.safe_load(f)
#     main(config_data)










# толкьо картинки
# from __future__ import annotations
# import sys
# import os
# sys.path.append(os.path.expanduser("~/work/python_libs"))

# import yaml, joblib, os
# import numpy as np
# from rich import print

# from src.datasets import load_jsonl, texts_images_labels
# from src.encoders import Blip2Encoder 
# from src.models import make_clf
# from src.utils import set_seed, get_device, ensure_dir

# def main(cfg):
#     set_seed(cfg.get("random_seed", 42))
#     # ВАЖНО: берем устройство из конфига
#     device = torch.device(cfg["blip2"].get("device", "cuda:0"))

#     train_data = load_jsonl(cfg["paths"]["train"])
#     val_data = load_jsonl(cfg["paths"]["val"])

#     tr_txt, tr_img, ytr = texts_images_labels(train_data)
#     va_txt, va_img, yva = texts_images_labels(val_data)

#     encoder = Blip2Encoder(cfg["blip2"]["name"], device)

#     print("[yellow]Encoding train (Unified BLIP-2)...[/yellow]")
#     Xtr = encoder.encode(tr_txt, tr_img, batch_size=cfg["blip2"]["batch_size"])
#     print("[yellow]Encoding val...[/yellow]")
#     Xva = encoder.encode(va_txt, va_img, batch_size=cfg["blip2"]["batch_size"])

#     clf = make_clf(kind=cfg["train"]["clf_type"], lr=float(cfg["train"]["lr"]))

#     best_acc = -1.0
#     patience = cfg["train"]["patience"]
#     wait = 0
#     ensure_dir(cfg["paths"]["outdir"])
#     save_path = os.path.join(cfg["paths"]["outdir"], "hybrid_model.joblib")

#     for epoch in range(1, cfg["train"]["epochs"] + 1):
#         clf.fit(Xtr, ytr)
#         acc = clf.score(Xva, yva)
#         print(f"Epoch {epoch:3d} | Val Acc: {acc:.4f} | Best: {best_acc:.4f}")

#         if acc > best_acc:
#             best_acc = acc
#             wait = 0
#             # Сохраняем только пайплайн
#             joblib.dump({"pipeline": clf}, save_path)
#             print("[green]Saved new best![/green]")
#         else:
#             wait += 1
#             if wait >= patience:
#                 print(f"[red]Early stopping at epoch {epoch}[/red]")
#                 break

# if __name__ == "__main__":
#     import torch
#     with open("config.yaml", "r") as f:
#         cfg = yaml.safe_load(f)
#     main(cfg)







from __future__ import annotations
import sys
import os
sys.path.append(os.path.expanduser("~/work/python_libs"))

import torch
import yaml, joblib
import numpy as np
from rich import print

from src.datasets import load_jsonl, texts_images_labels
from src.encoders import Blip2Encoder 
from src.models import make_clf
from src.utils import set_seed, ensure_dir

def main(cfg):
    set_seed(cfg.get("random_seed", 42))
    
    # Явное создание torch.device
    device = torch.device(cfg["blip2"].get("device", "cuda:0"))

    train_data = load_jsonl(cfg["paths"]["train"])
    val_data = load_jsonl(cfg["paths"]["val"])

    tr_txt, tr_img, ytr = texts_images_labels(train_data)
    va_txt, va_img, yva = texts_images_labels(val_data)

    encoder = Blip2Encoder(cfg["blip2"]["name"], device)

    print("[yellow]Encoding train (Multimodal Fusion)...[/yellow]")
    Xtr = encoder.encode(tr_txt, tr_img, batch_size=cfg["blip2"]["batch_size"])
    print("[yellow]Encoding val...[/yellow]")
    Xva = encoder.encode(va_txt, va_img, batch_size=cfg["blip2"]["batch_size"])

    # Собираем пайплайн (теперь он включает Normalizer)
    clf = make_clf(kind=cfg["train"]["clf_type"], lr=float(cfg["train"]["lr"]))

    best_acc = -1.0
    patience = cfg["train"]["patience"]
    wait = 0
    ensure_dir(cfg["paths"]["outdir"])
    save_path = os.path.join(cfg["paths"]["outdir"], "hybrid_model.joblib")

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        clf.fit(Xtr, ytr)
        acc = clf.score(Xva, yva)
        print(f"Epoch {epoch:3d} | Val Acc: {acc:.4f} | Best: {best_acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            wait = 0
            # Сохраняем весь пайплайн целиком
            joblib.dump({"pipeline": clf}, save_path)
            print("[green]Saved new best multimodal model![/green]")
        else:
            wait += 1
            if wait >= patience:
                print(f"[red]Early stopping. Best Acc: {best_acc:.4f}[/red]")
                break

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)