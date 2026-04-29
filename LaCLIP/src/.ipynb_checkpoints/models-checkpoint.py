# from __future__ import annotations

# import sys
# import os
# sys.path.append(os.path.expanduser("~/work/python_libs"))

# import numpy as np
# from sklearn.linear_model import LogisticRegression
# from sklearn.neural_network import MLPClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline

# def make_clf(kind: str = "logistic", normalize: bool = True, lr: float = 1e-3, weight_decay: float = 0.0, epochs: int = 20, **kwargs):
#     steps = []
    
#     # 1. Добавляем нормализацию, если нужно
#     if normalize:
#         steps.append(("scaler", StandardScaler()))
    
#     # 2. Настраиваем саму модель
#     if kind == "logistic":
#         # Передаем kwargs (там будет наш warm_start) прямо в логистическую регрессию
#         clf = LogisticRegression(max_iter=4000, n_jobs=-1, **kwargs)
#         steps.append(("clf", clf))
        
#     elif kind == "mlp":
#         # Передаем kwargs в MLP
#         # Важно: если мы сами крутим цикл эпох в train_multi, 
#         # лучше выключить внутренний early_stopping самого sklearn
#         mlp = MLPClassifier(
#             hidden_layer_sizes=(256,), 
#             activation='relu', 
#             batch_size=128, 
#             learning_rate_init=lr,
#             max_iter=epochs, 
#             verbose=False, 
#             **kwargs
#         )
#         steps.append(("clf", mlp))
#     else:
#         raise ValueError(f"Unknown clf type: {kind}")

#     return Pipeline(steps)


from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class TorchMLP(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim=1025, hidden_dim=256, lr=1e-4, dropout=0.7, weight_decay=0.05):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.dropout = dropout
        self.weight_decay = weight_decay
        
        # Инициализируем архитектуру
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(hidden_dim, 2)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        self.model.train()
        X_t = torch.as_tensor(X, dtype=torch.float32)
        y_t = torch.as_tensor(y, dtype=torch.long)
        
        self.optimizer.zero_grad()
        outputs = self.model(X_t)
        loss = self.criterion(outputs, y_t)
        loss.backward()
        self.optimizer.step()
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.as_tensor(X, dtype=torch.float32)
            logits = self.model(X_t)
            return torch.argmax(logits, dim=1).numpy()

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.as_tensor(X, dtype=torch.float32)
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=1)
            return probs.numpy()

def make_clf(kind: str = "mlp", normalize: bool = True, lr: float = 1e-4, weight_decay: float = 0.05, **kwargs):
    steps = []
    if normalize:
        steps.append(("scaler", StandardScaler()))
    
    if kind == "mlp":
        # Убедитесь, что input_dim соответствует размеру ваших признаков
        clf = TorchMLP(input_dim=1025, lr=lr, dropout=0.7, weight_decay=weight_decay)
        steps.append(("clf", clf))
    else:
        from sklearn.linear_model import LogisticRegression
        steps.append(("clf", LogisticRegression(max_iter=4000, **kwargs)))
        
    return Pipeline(steps)