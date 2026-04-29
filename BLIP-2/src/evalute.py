from __future__ import annotations
import sys
import os
sys.path.append(os.path.expanduser("~/work/python_libs"))

import torch
import joblib
import argparse
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from rich import print
from rich.table import Table

from src.datasets import load_jsonl, texts_images_labels
from src.encoders import Blip2Encoder
from src.utils import get_device

def main():
    parser = argparse.ArgumentParser()
    # Укажи путь к тестовому или валидационному файлу
    parser.add_argument("--data", type=str, default="../data/val.jsonl")
    parser.add_argument("--model", type=str, default="outputs/hybrid_model.joblib")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    device = get_device("auto")
    
    # 1. Загрузка модели и данных
    print(f"[bold blue]Loading model:[/bold blue] {args.model}")
    model_data = joblib.load(args.model)
    
    # Инициализируем энкодер (BLIP-2 + SBERT)
    encoder = Blip2Encoder("Salesforce/blip2-opt-2.7b", device)

    print(f"[bold blue]Loading data:[/bold blue] {args.data}")
    dataset = load_jsonl(args.data)
    texts, images, y_true = texts_images_labels(dataset)

    # 2. Получение эмбеддингов
    print(f"[yellow]Encoding {len(texts)} samples...[/yellow]")
    X = encoder.encode(texts, images, batch_size=args.batch_size)
    X = np.nan_to_num(X.astype(np.float32))

    # 3. Предсказание
    clf = model_data["pipeline"]
    y_pred = clf.predict(X)
    
    # Если модель поддерживает вероятности, можно вытащить уверенность
    y_probs = clf.predict_proba(X)

    # 4. Расчет и вывод метрик
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)

    # Красивая таблица результатов через rich
    table = Table(title=f"Model Evaluation Metrics (Acc: {acc:.4f})")
    table.add_column("Class", justify="left", style="cyan")
    table.add_column("Precision", justify="right", style="magenta")
    table.add_column("Recall", justify="right", style="magenta")
    table.add_column("F1-Score", justify="right", style="green")

    for label, metrics in report.items():
        if label in ['accuracy', 'macro avg', 'weighted avg']:
            continue
        table.add_row(
            str(label),
            f"{metrics['precision']:.3f}",
            f"{metrics['recall']:.3f}",
            f"{metrics['f1-score']:.3f}"
        )

    print("-" * 50)
    print(table)
    print("-" * 50)
    
    # Матрица ошибок (Confusion Matrix) для понимания, где модель ошибается
    cm = confusion_matrix(y_true, y_pred)
    print(f"[bold]Confusion Matrix:[/bold]\n{cm}")

if __name__ == "__main__":
    main()