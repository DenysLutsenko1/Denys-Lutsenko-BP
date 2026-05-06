from __future__ import annotations
import sys
import os
sys.path.append(os.path.expanduser("~/work/python_libs"))

import torch
import joblib
import argparse
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from rich import print as rprint

from src.datasets import load_jsonl, texts_images_labels
from src.encoders import Blip2Encoder
from src.utils import get_device

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../data/val.jsonl")
    parser.add_argument("--model", "--ckpt", type=str, dest="model", default="outputs/hybrid_model.joblib")
    parser.add_argument("--config", type=str)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    device = get_device("auto")
    
    rprint(f"[bold blue]Loading model:[/bold blue] {args.model}")
    model_data = joblib.load(args.model)
    
    encoder = Blip2Encoder("Salesforce/blip2-opt-2.7b", device)

    rprint(f"[bold blue]Loading data:[/bold blue] {args.data}")
    dataset = load_jsonl(args.data)
    texts, images, y_true = texts_images_labels(dataset)

    rprint(f"[yellow]Encoding {len(texts)} samples...[/yellow]")
    X = encoder.encode(texts, images, batch_size=args.batch_size)
    X = np.nan_to_num(X.astype(np.float32))

    clf = model_data["pipeline"]
    y_pred = clf.predict(X)
    y_probs = clf.predict_proba(X)

    # --- ИСПРАВЛЕННЫЙ РАСЧЕТ ROC AUC ---
    try:
        # Если классов всего 2 (shape[1] == 2), берем только вероятности класса 1
        if y_probs.shape[1] == 2:
            roc_auc = roc_auc_score(y_true, y_probs[:, 1])
        else:
            # Для многоклассовой классификации используем взвешенный OVR
            roc_auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='weighted')
    except Exception as e:
        rprint(f"[bold red]ROC AUC Error:[/bold red] {e}")
        roc_auc = 0.0

    # Получаем F1 Score (weighted) для вывода в заголовке
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    f1_weighted = report_dict['weighted avg']['f1-score']

    # --- ВЫВОД В СТИЛЕ ВАШЕГО СКРИНШОТА ---
    print("\n" + "="*45)
    print("             TESTSET RESULTS             ")
    print("="*45)
    print(f"ROC AUC:  {roc_auc:.4f}")
    print(f"F1 Score: {f1_weighted:.4f}")
    print("-" * 45)
    
    # digits=4 выведет 0.9524 вместо 0.95
    print(classification_report(y_true, y_pred, digits=4))

if __name__ == "__main__":
    main()