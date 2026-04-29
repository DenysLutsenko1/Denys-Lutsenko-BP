# from __future__ import annotations
# import sys
# import os
# sys.path.append(os.path.expanduser("~/work/python_libs"))

# from sklearn.neural_network import MLPClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler


# def make_clf(
#     kind: str = "mlp",
#     normalize: bool = False, # ОТКЛЮЧЕНО ДЛЯ ТЕСТОВ
#     lr: float = 1e-3,
#     epochs: int = 1,
#     **kwargs
# ):
#     steps = []

#     if normalize:
#         steps.append(("scaler", StandardScaler()))

#     if kind == "mlp":
#         # Вход 1152 (768 BLIP + 384 SBERT)
#         mlp = MLPClassifier(
#             hidden_layer_sizes=(512, 128),
#             activation="relu",
#             batch_size="auto",
#             learning_rate_init=lr,
#             max_iter=epochs,
#             verbose=False,
#             early_stopping=False,
#             **kwargs
#         )
#         steps.append(("clf", mlp))

#     elif kind == "logistic":
#         clf = LogisticRegression(max_iter=4000, n_jobs=-1, **kwargs)
#         steps.append(("clf", clf))

#     return Pipeline(steps)

# толкьо картики
# from __future__ import annotations
# import sys
# import os
# sys.path.append(os.path.expanduser("~/work/python_libs"))

# from sklearn.neural_network import MLPClassifier
# from sklearn.pipeline import Pipeline

# def make_clf(kind: str = "mlp", lr: float = 0.0005, **kwargs):
#     steps = []
#     if kind == "mlp":
#         mlp = MLPClassifier(
#             hidden_layer_sizes=(512, 256, 128),
#             activation="relu",
#             solver="adam",
#             alpha=0.01,
#             learning_rate_init=lr,
#             max_iter=1,
#             warm_start=True,
#             **kwargs
#         )
#         steps.append(("clf", mlp))
#     return Pipeline(steps)






from __future__ import annotations
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

def make_clf(kind: str = "mlp", lr: float = 0.001, **kwargs):
    steps = []
    # Важно для мультимодальных векторов: нормализуем их перед MLP
    steps.append(("norm", Normalizer())) 
    
    if kind == "mlp":
        mlp = MLPClassifier(
            hidden_layer_sizes=(512, 256),
            activation="relu",
            solver="adam",
            alpha=0.001, # Немного уменьшили регуляризацию для чувствительности
            learning_rate_init=lr,
            max_iter=1,
            warm_start=True,
            **kwargs
        )
        steps.append(("clf", mlp))
    return Pipeline(steps)