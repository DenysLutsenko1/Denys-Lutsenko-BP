from __future__ import annotations
from typing import List, Literal
import sys
import os
sys.path.append(os.path.expanduser("~/work/python_libs"))





# 1. Отключаем проверку безопасности на уровне системы, чтобы разрешить загрузку весов
os.environ["HF_HUB_DISABLE_TORCH_SAFE_LOAD_CHECK"] = "1"

import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoConfig, 
    CLIPTokenizer, 
    CLIPTextModel,
    CLIPProcessor, 
    CLIPModel,
    ViltModel,
    ViltProcessor
)
from PIL import Image

@torch.no_grad()
def pool_hidden(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    mode: Literal["mean", "max", "cls"] = "mean",
) -> torch.Tensor:
    """Объединяет скрытые состояния токенов в один вектор предложения."""
    if mode == "cls":
        return hidden[:, 0]

    mask = attention_mask.unsqueeze(-1).type_as(hidden)
    hidden = hidden * mask

    if mode == "mean":
        summed = hidden.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts

    if mode == "max":
        hidden = hidden.masked_fill(mask == 0, -1e9)
        return hidden.max(dim=1).values

    raise ValueError(f"Unknown pool mode: {mode!r}. Expected one of: 'mean', 'max', 'cls'.")

class TextEncoder:
    """Универсальный текстовый энкодер (CLIP, BERT и др.) с поддержкой CUDA и Safetensors."""

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        max_length: int = 128,
        pool: str = "mean",
    ):
        self.model_name = model_name
        self.device = device
        self.pool = pool

        # Загружаем конфиг с явным указанием безопасного формата
        cfg = AutoConfig.from_pretrained(model_name, use_safetensors=True)
        self._is_clip = getattr(cfg, "model_type", None) == "clip"

        if self._is_clip:
            self.tok = CLIPTokenizer.from_pretrained(model_name)
            # Загружаем текстовую часть CLIP через Safetensors
            self.model = CLIPTextModel.from_pretrained(model_name, use_safetensors=True)
        else:
            self.tok = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name, use_safetensors=True)

        self.model.eval().to(device)

        tok_limit = getattr(self.tok, "model_max_length", max_length)
        self.max_length = min(
            max_length,
            tok_limit if tok_limit and tok_limit > 0 else max_length,
        )

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tok(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            out = self.model(**inputs)

            if self._is_clip and hasattr(out, "pooler_output") and out.pooler_output is not None:
                pooled = out.pooler_output
            else:
                pooled = pool_hidden(out.last_hidden_state, inputs["attention_mask"], mode=self.pool)

            embs.append(pooled.detach().cpu().numpy())

        return np.vstack(embs)

class ImageEncoder:
    """Энкодер изображений CLIP с поддержкой CUDA и Safetensors."""

    def __init__(self, model_name: str, device: torch.device):
        self.device = device
        # Загружаем модель CLIP целиком через Safetensors и сразу на GPU
        self.model = CLIPModel.from_pretrained(model_name, use_safetensors=True).eval().to(device)
        self.proc = CLIPProcessor.from_pretrained(model_name)

    def _encode_batch(self, imgs: List) -> np.ndarray:
        inputs = self.proc(images=imgs, return_tensors="pt").to(self.device)
        out = self.model.get_image_features(pixel_values=inputs["pixel_values"])
        
        if isinstance(out, torch.Tensor):
            feats = out
        elif hasattr(out, "image_embeds"):
            feats = out.image_embeds
        elif hasattr(out, "pooler_output"):
            feats = out.pooler_output
        else:
            raise AttributeError(f"Unexpected output type: {type(out)}")
        
        return feats.detach().cpu().numpy()

    @torch.no_grad()
    def encode_paths(self, image_paths: List[str], batch_size: int = 32) -> np.ndarray:
        embs = []
        batch_imgs: List = []

        for p in image_paths:
            batch_imgs.append(Image.open(p).convert("RGB"))

            if len(batch_imgs) == batch_size:
                embs.append(self._encode_batch(batch_imgs))
                batch_imgs = []

        if batch_imgs:
            embs.append(self._encode_batch(batch_imgs))

        return np.vstack(embs)

class ViltEncoder:
    """Энкодер ViLT для мультимодального извлечения признаков."""

    def __init__(self, model_name: str, device: torch.device):
        self.device = device
        # Загружаем ViLT через Safetensors и на GPU
        self.model = ViltModel.from_pretrained(model_name, use_safetensors=True).eval().to(device)
        self.proc = ViltProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def encode(self, texts: List[str], images: List[str | None], batch_size: int = 32) -> np.ndarray:
        embs = []
        for i in range(0, len(texts), batch_size):
            b_texts = texts[i : i + batch_size]
            b_imgs = []
            for img_path in images[i : i + batch_size]:
                if img_path and os.path.exists(img_path):
                    b_imgs.append(Image.open(img_path).convert("RGB"))
                else:
                    # Если картинки нет, создаем пустой черный квадрат
                    b_imgs.append(Image.new("RGB", (224, 224), (0, 0, 0)))

            inputs = self.proc(images=b_imgs, text=b_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            out = self.model(**inputs)
            
            # Используем pooler_output (финальный вектор [CLS] токена)
            pooled = out.pooler_output
            embs.append(pooled.detach().cpu().numpy())

        return np.vstack(embs)