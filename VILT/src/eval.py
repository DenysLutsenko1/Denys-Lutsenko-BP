from __future__ import annotations

import sys
import os
# Добавляем путь к библиотекам
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

    # Извлекаем сохраненный лимит токенов
    max_len = bundle.get('max_length', cfg['vilt'].get('max_length', 40))
    print(f"Context length: {max_len}")

    # ВАЖНО: Фиксируем сид перед инициализацией энкодера
    set_seed(42)

    # 3. Инициализация ViLT энкодера
    encoder = ViltEncoder(bundle['model_name'], device, max_length=max_len)

    # 4. Кодирование
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

    # --- РАСЧЕТ МЕТРИК ДЛЯ ШАПКИ ---
    try:
        auc = roc_auc_score(yte, prob)
    except:
        auc = 0.0

    report_dict = classification_report(yte, pred, output_dict=True)
    f1_weighted = report_dict['weighted avg']['f1-score']

    # --- КРАСИВЫЙ ВЫВОД (КАК НА СКРИНШОТЕ) ---
    print("\n" + "="*45)
    print("             TESTSET RESULTS             ")
    print("="*45)
    print(f"ROC AUC:  {auc:.4f}")
    print(f"F1 Score: {f1_weighted:.4f}")
    print("-" * 45)
    
    # Печатаем стандартный текстовый отчет sklearn с 4 знаками
    report = classification_report(yte, pred, digits=4)
    print(report)
    print("="*45)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='config.yaml')
    ap.add_argument('--ckpt', type=str, required=True, help="Путь к multimodal_clf.joblib")
    ap.add_argument('--threshold', type=float, default=0.5)
    args = ap.parse_args()
    
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
        
    main(cfg, args.ckpt, args.threshold)