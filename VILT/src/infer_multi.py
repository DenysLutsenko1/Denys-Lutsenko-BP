# from __future__ import annotations

# import sys
# import os
# sys.path.append(os.path.expanduser("~/work/python_libs"))


# import argparse, joblib, yaml, os
# import numpy as np
# from src.encoders import ViltEncoder
# from src.utils import get_device, set_seed # Добавили set_seed

# def infer(bundle_path: str, texts: list[str], image_paths: list[str], device_pref: str = "auto"):
#     """
#     Инференс для ViLT: принимает списки текстов и путей к фото.
#     """
#     # 1. Загрузка бандла и устройства
#     bundle = joblib.load(bundle_path)
#     device = get_device(device_pref)

#     # 2. Извлекаем сохраненный max_length
#     max_len = bundle.get("max_length", 40)
#     print(f"[*] Model loaded with max_length context: {max_len}")

#     # ВАЖНО: Фиксируем сид ПЕРЕД инициализацией энкодера.
#     # Это гарантирует, что MISMATCH-веса (от 41 до 512) всегда будут одинаковыми.
#     set_seed(42) 

#     # 3. Инициализация мультимодального энкодера с нужным лимитом
#     encoder = ViltEncoder(bundle['model_name'], device, max_length=max_len)

#     # 4. Кодирование
#     print(f"Encoding {len(texts)} samples with ViLT...")
#     X = encoder.encode(texts, image_paths, batch_size=len(texts))

#     # 5. Предсказание классификатором
#     clf = bundle["pipeline"]
#     prob = clf.predict_proba(X)[:, 1]
    
#     return prob

# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--ckpt", type=str, required=True, help="Путь к multimodal_clf.joblib")
#     ap.add_argument("--text", type=str, nargs="+", required=True)
#     ap.add_argument("--image", type=str, nargs="+", required=True)
#     ap.add_argument("--device", type=str, default="auto")
#     args = ap.parse_args()

#     if len(args.text) != len(args.image):
#         print("Ошибка: Количество текстов и картинок должно совпадать!")
#     else:
#         # Устанавливаем сид и здесь на всякий случай
#         set_seed(42)
#         results = infer(args.ckpt, args.text, args.image, args.device)
        
#         print("\n=== Predictions ===")
#         for i, p in enumerate(results):
#             label = "AI (ИИ)" if p >= 0.5 else "HUMAN (Человек)"
#             print(f"Sample {i+1}: Prob={p:.4f} -> {label}")

from __future__ import annotations

import sys
import os
sys.path.append(os.path.expanduser("~/work/python_libs"))

import argparse, joblib, yaml
import numpy as np
import torch
from src.encoders import ViltEncoder
from src.utils import get_device, set_seed

def infer(bundle_path: str, texts: list[str], image_paths: list[str], device_pref: str = "auto"):
    """
    Инференс для ViLT: принимает списки текстов и путей к фото.
    """
    # 1. Загрузка бандла
    bundle = joblib.load(bundle_path)
    
    # Определяем устройство (лучше явно указать cuda:1, если на ней учил)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # 2. Извлекаем сохраненный max_length
    max_len = bundle.get("max_length", 40)
    print(f"[*] Model loaded with max_length context: {max_len}")

    # Фиксируем сид для воспроизводимости весов при инициализации расширенного контекста
    set_seed(42) 

    # 3. Инициализация мультимодального энкодера
    # Важно: используем то же имя модели, что было при обучении
    encoder = ViltEncoder(bundle['model_name'], device, max_length=max_len)

    # 4. Кодирование
    # ОБЯЗАТЕЛЬНО: dropout_prob=0.0, так как это предсказание, а не обучение
    print(f"Encoding {len(texts)} samples with ViLT...")
    X = encoder.encode(texts, image_paths, batch_size=len(texts), dropout_prob=0.0)

    # 5. Предсказание классификатором
    clf = bundle["pipeline"]
    prob = clf.predict_proba(X)[:, 1]
    
    return prob

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Путь к multimodal_clf.joblib")
    ap.add_argument("--text", type=str, nargs="+", required=True)
    ap.add_argument("--image", type=str, nargs="+", required=True)
    args = ap.parse_args()

    if len(args.text) != len(args.image):
        print("Ошибка: Количество текстов и картинок должно совпадать!")
    else:
        set_seed(42)
        results = infer(args.ckpt, args.text, args.image)
        
        print("\n=== Predictions ===")
        for i, p in enumerate(results):
            label = "AI (ИИ)" if p >= 0.5 else "HUMAN (Человек)"
            print(f"Sample {i+1}: Prob={p:.4f} -> {label}")