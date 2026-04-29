# from __future__ import annotations

# import sys
# import os
# sys.path.append(os.path.expanduser("~/work/python_libs"))


# import argparse, yaml, joblib, torch
# import numpy as np
# from sklearn.metrics import classification_report, roc_auc_score
# from rich import print # Добавил rich для красивого вывода, как в твоем train_multi

# # Используем новую чистую загрузку из datasets.py
# from src.datasets import load_jsonl, texts_images_labels
# from src.encoders import ViltEncoder
# from src.utils import get_device

# def main(cfg, ckpt, threshold: float):
#     # 1. Загрузка данных
#     test_data = load_jsonl(cfg['paths']['test'])
#     t_txt, t_img, yte = texts_images_labels(test_data)

#     device = get_device(cfg['vilt'].get('device', 'auto'))
    
#     # Загружаем бандл модели
#     print(f"Loading model bundle from: {ckpt}")
#     bundle = joblib.load(ckpt)

#     # ВАЖНО: Извлекаем сохраненный max_length из бандла
#     # Если его там нет (старая модель), откатываемся к значению из конфига или 40
#     max_len = bundle.get('max_length', cfg['vilt'].get('max_length', 40))
#     print(f"[bold blue]>>> Context length for evaluation:[/bold blue] {max_len}")

#     # 2. Инициализация ViLT энкодера с правильным max_length
#     # Теперь энкодер расширит позиционную матрицу так же, как при обучении
#     encoder = ViltEncoder(bundle['model_name'], device, max_length=max_len)

#     # 3. Кодирование тестовых данных
#     print(f"Encoding {len(t_txt)} test samples with ViLT...")
#     X_test = encoder.encode(t_txt, t_img, batch_size=cfg['vilt']['batch_size'])

#     # 4. Классификация
#     clf = bundle['pipeline']
    
#     # Получаем вероятности
#     prob = clf.predict_proba(X_test)[:, 1]
#     # Применяем порог (threshold)
#     pred = (prob >= threshold).astype(int)

#     # 5. Метрики
#     print(f"\n[bold green]Evaluation Results (Threshold: {threshold}):[/bold green]")
#     try:
#         auc = roc_auc_score(yte, prob)
#         print(f"ROC AUC: {auc:.4f}")
#     except Exception as e:
#         print(f"AUC Error: {e}")
    
#     # Выводим отчет классификации
#     print(classification_report(yte, pred, digits=4))

# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument('--config', type=str, default='config.yaml')
#     ap.add_argument('--ckpt', type=str, required=True, help="Путь к multimodal_clf.joblib")
#     ap.add_argument('--threshold', type=float, default=0.5)
#     args = ap.parse_args()
    
#     with open(args.config, 'r', encoding='utf-8') as f:
#         cfg = yaml.safe_load(f)
        
#     main(cfg, args.ckpt, args.threshold)

from __future__ import annotations

import sys
import os
# Добавляем путь к библиотекам, если нужно
sys.path.append(os.path.expanduser("~/work/python_libs"))

import argparse, yaml, joblib, torch
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from rich import print

# Импорты твоих модулей
from src.datasets import load_jsonl, texts_images_labels
from src.encoders import ViltEncoder
from src.utils import get_device, set_seed

def main(cfg, ckpt, threshold: float):
    # 1. Загрузка тестовых данных
    test_data = load_jsonl(cfg['paths']['test'])
    t_txt, t_img, yte = texts_images_labels(test_data)

    # Используем ту же карту, что и при обучении
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    # 2. Загрузка бандла модели
    print(f"Loading model bundle from: {ckpt}")
    bundle = joblib.load(ckpt)

    # Извлекаем сохраненный лимит токенов (512)
    max_len = bundle.get('max_length', cfg['vilt'].get('max_length', 40))
    print(f"Context length: {max_len}")

    # ВАЖНО: Фиксируем сид перед инициализацией энкодера, 
    # чтобы веса для длинного текста (41-512) совпали с тренировочными
    set_seed(42)

    # 3. Инициализация ViLT энкодера
    encoder = ViltEncoder(bundle['model_name'], device, max_length=max_len)

    # 4. Кодирование (Dropout всегда 0.0 на тесте)
    print(f"Encoding {len(t_txt)} test samples...")
    X_test = encoder.encode(
        t_txt, 
        t_img, 
        batch_size=cfg['vilt']['batch_size'], 
        dropout_prob=0.0
    )

    # 5. Классификация
    clf = bundle['pipeline']
    prob = clf.predict_proba(X_test)[:, 1]
    pred = (prob >= threshold).astype(int)

    # 6. Вывод результатов (без ошибок разметки rich)
    print("\n" + "="*30)
    print("      TESTSET RESULTS      ")
    print("="*30)
    
    try:
        auc = roc_auc_score(yte, prob)
        print(f"ROC AUC: {auc:.4f}")
    except Exception as e:
        print(f"AUC calculation error: {e}")
    
    print(f"Threshold: {threshold}")
    print("-" * 30)
    
    # Печатаем стандартный текстовый отчет sklearn
    report = classification_report(yte, pred, digits=4)
    print(report)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='config.yaml')
    ap.add_argument('--ckpt', type=str, required=True, help="Путь к multimodal_clf.joblib")
    ap.add_argument('--threshold', type=float, default=0.5)
    args = ap.parse_args()
    
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
        
    main(cfg, args.ckpt, args.threshold)