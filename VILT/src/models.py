from __future__ import annotations

import sys
import os
sys.path.append(os.path.expanduser("~/work/python_libs"))



import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def make_clf(
    kind: str = "mlp", 
    normalize: bool = True, 
    lr: float = 1e-3, 
    epochs: int = 1, 
    **kwargs
):
    """
    Создает классификатор для работы с эмбеддингами ViLT (768 признаков).
    """
    steps = []
    
    # 1. Нормализация
    # Для MLP это критично, так как эмбеддинги могут иметь разный масштаб
    if normalize:
        steps.append(("scaler", StandardScaler()))
    
    # 2. Выбор классификатора
    if kind == "logistic":
        clf = LogisticRegression(
            max_iter=4000, 
            n_jobs=-1, 
            **kwargs
        )
        steps.append(("clf", clf))
        
    elif kind == "mlp":
        # Увеличиваем архитектуру для более сложных зависимостей длинного текста
        mlp = MLPClassifier(
            hidden_layer_sizes=(512, 256), # Для ViLT достаточно двух слоев
            activation='relu', 
            solver='adam',
            alpha=0.01,         # Добавляем регуляризацию, чтобы не переобучаться на слова
            batch_size=128, 
            learning_rate_init=lr,
            max_iter=epochs, 
            warm_start=kwargs.get('warm_start', True)
            # hidden_layer_sizes=(512, 256, 128), # Добавили слой 512 для лучшей емкости
            # activation='relu', 
            # solver='adam',      # Явно указываем оптимизатор
            # batch_size=128, 
            # learning_rate_init=lr,
            # max_iter=epochs, 
            # verbose=False, 
            # early_stopping=False,
            # **kwargs              # Принимает warm_start=True из train_multi.py
        )
        steps.append(("clf", mlp))
    else:
        raise ValueError(f"Unknown clf type: {kind}. Use 'mlp' or 'logistic'.")

    return Pipeline(steps)