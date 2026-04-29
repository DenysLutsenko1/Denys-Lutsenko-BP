from __future__ import annotations
# import sys
# import os
# sys.path.append(os.path.expanduser("~/work/python_libs"))

# # import torch
# # import joblib
# # import argparse
# # import numpy as np
# # from PIL import Image
# # from rich import print

# # from src.encoders import Blip2Encoder
# # from src.utils import get_device

# # def predict_one(image_path: str, text: str, encoder: Blip2Encoder, model_data: dict):
# #     # 1. Получаем вектор (эмбеддинг) для этой пары
# #     # Метод encode ожидает списки, поэтому оборачиваем в []
# #     print(f"[yellow]Encoding input...[/yellow]")
# #     embedding = encoder.encode([text], [image_path], batch_size=1)
    
# #     # Защита от NaN
# #     embedding = np.nan_to_num(embedding.astype(np.float32))

# #     # 2. Предсказание
# #     clf = model_data["pipeline"]
# #     # Находим индекс самого вероятного класса
# #     probs = clf.predict_proba(embedding)[0]
# #     pred_class = clf.predict(embedding)[0]
    
# #     return pred_class, probs

# # def main():
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("--image", type=str, required=True, help="Путь к картинке")
# #     parser.add_argument("--text", type=str, required=True, help="Текст описания")
# #     parser.add_argument("--model", type=str, default="outputs/hybrid_model.joblib")
# #     args = parser.parse_args()

# #     device = get_device("auto")
    
# #     # Загружаем сохраненные веса классификатора
# #     print(f"[bold blue]Loading model weights:[/bold blue] {args.model}")
# #     model_data = joblib.load(args.model)

# #     # Инициализируем те же энкодеры (нужно имя модели из конфига, по умолчанию Salesforce/blip2-opt-2.7b)
# #     encoder = Blip2Encoder("Salesforce/blip2-opt-2.7b", device)

# #     # Запускаем тест
# #     label, probs = predict_one(args.image, args.text, encoder, model_data)

# #     print("-" * 30)
# #     print(f"[bold green]RESULT:[/bold green]")
# #     print(f"Text: {args.text}")
# #     print(f"Image: {args.image}")
# #     print(f"Predicted Class: [bold cyan]{label}[/bold cyan]")
# #     print(f"Confidence: [yellow]{np.max(probs):.4f}[/yellow]")
# #     print("-" * 30)

# # if __name__ == "__main__":
# #     main()

# import torch
# import joblib
# import argparse
# import numpy as np
# import os
# from rich import print

# from src.encoders import Blip2Encoder
# from src.utils import get_device

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--image", type=str, required=True)
#     parser.add_argument("--text", type=str, required=True)
#     parser.add_argument("--model", type=str, default="outputs/hybrid_model.joblib")
#     args = parser.parse_args()

#     device = get_device("auto")
    
#     # Загружаем модель
#     model_data = joblib.load(args.model)
#     encoder = Blip2Encoder("Salesforce/blip2-opt-2.7b", device)

#     # 1. Получаем эмбеддинг
#     embedding = encoder.encode([args.text], [args.image], batch_size=1)
#     embedding = np.nan_to_num(embedding.astype(np.float32))

#     # 2. Считаем вероятности
#     clf = model_data["pipeline"]
#     # predict_proba возвращает список вероятностей для каждого класса
#     # Допустим, класс 1 — это "сгенерировано" (genai)
#     probs = clf.predict_proba(embedding)[0]
    
#     # Берем вероятность для второго класса (индекс 1)
#     # Если у тебя только один класс или другой порядок, можно подправить индекс
#     p_genai = probs[1] if len(probs) > 1 else probs[0]

#     # 3. Вывод в формате как на скриншоте
#     # Форматируем путь, чтобы он был коротким (относительным)
#     short_img_path = os.path.relpath(args.image, start="/home/jovyan/work/bp_latex/")
    
#     print(f"00 p(genai)={p_genai:.3f}  text='{args.text}'  image='{short_img_path}'")

# if __name__ == "__main__":
#     main()

import torch
import joblib
import argparse
import numpy as np
import os
import sys
from rich import print

# Путь к библиотекам
sys.path.append(os.path.expanduser("~/work/python_libs"))

from src.encoders import Blip2Encoder
from src.utils import get_device

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--model", "--ckpt", type=str, default="outputs/hybrid_model.joblib", dest="model_path")
    args = parser.parse_args()

    # ВАЖНО: используем ту же карту, что при обучении (или cuda:0 если через Visible Devices)
    device = torch.device("cuda:0")
    
    # 1. Загрузка модели
    model_data = joblib.load(args.model_path)
    pipeline = model_data["pipeline"]
    
    # 2. Инициализация энкодера (нового, без SBERT)
    encoder = Blip2Encoder("Salesforce/blip2-opt-2.7b", device)

    # 3. Получение эмбеддинга
    embedding = encoder.encode([args.text], [args.image], batch_size=1)
    
    # 4. Предикт (БЕЗ скейлера!)
    probs = pipeline.predict_proba(embedding)[0]
    
    # Определяем индекс класса GenAI (обычно 1)
    # Если в обучении 1 был GenAI, то берем индекс для 1
    idx = 1 if pipeline.classes_[1] == 1 else 0
    p_genai = probs[idx]

    print(f"\n[bold green]RESULT:[/bold green]")
    print(f"P(Generated) = {p_genai:.4f}")
    print(f"P(Real)      = {1 - p_genai:.4f}")
    
    if p_genai > 0.5:
        print("[red]Verdict: AI GENERATED[/red]")
    else:
        print("[cyan]Verdict: REAL PHOTO[/cyan]")

if __name__ == "__main__":
    main()