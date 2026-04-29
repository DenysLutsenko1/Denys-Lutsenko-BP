from __future__ import annotations
import sys
import os
sys.path.append(os.path.expanduser("~/work/python_libs"))
import torch
import inspect
from transformers import Blip2ForConditionalGeneration

model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", 
    torch_dtype=torch.float16
)

print("\n" + "="*50)
print("1. Ключи, которые принимает Q-Former:")
sig = inspect.signature(model.qformer.forward)
print(list(sig.parameters.keys()))

print("\n2. Ключи, которые принимает BERT внутри Q-Former:")
sig_bert = inspect.signature(model.qformer.bert.forward)
print(list(sig_bert.parameters.keys()))

print("\n3. Есть ли у модели текстовый энкодер на верхнем уровне?")
print("get_text_features:", hasattr(model, "get_text_features"))
print("="*50 + "\n")