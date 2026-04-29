from __future__ import annotations

import sys
import os
sys.path.append(os.path.expanduser("~/work/python_libs"))


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline



def make_clf(kind: str = "logistic", normalize: bool = True, lr: float = 1e-3, weight_decay: float = 0.0, epochs: int = 20, **kwargs):
    steps = []
    if normalize:
        steps.append(("scaler", StandardScaler()))
    
    if kind == "logistic":
        # Добавляем **kwargs сюда
        steps.append(("clf", LogisticRegression(max_iter=4000, n_jobs=-1, **kwargs)))
        return Pipeline(steps)
    
    elif kind == "mlp":
        # Добавляем **kwargs сюда
        mlp = MLPClassifier(
            hidden_layer_sizes=(256,128), 
            solver='adam',
            alpha=0.015,
            activation='relu', 
            batch_size=128, 
            learning_rate_init=lr,
            max_iter=epochs, 
            verbose=False, 
            **kwargs # Это позволит передать warm_start=True
        )
        steps.append(("clf", mlp))
        return Pipeline(steps)
    else:
        raise ValueError("Unknown clf type")