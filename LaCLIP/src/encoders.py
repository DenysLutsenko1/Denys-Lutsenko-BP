from __future__ import annotations

import sys
import os
sys.path.append(os.path.expanduser("~/work/python_libs"))



from typing import List, Literal, Optional
import os

# 1. Отключаем проверку безопасности на уровне системы
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
    mode: Literal["mean","max","cls"]="mean"
) -> torch.Tensor:
    """Объединение скрытых состояний (для BERT-подобных моделей)."""
    if mode == "cls":
        return hidden[:, 0]
    mask = attention_mask.unsqueeze(-1).type_as(hidden)
    hidden = hidden * mask
    if mode == "mean":
        summed = hidden.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts
    elif mode == "max":
        hidden = hidden.masked_fill(mask == 0, -1e9)
        return hidden.max(dim=1).values
    raise ValueError("Unknown pool mode")

class TextEncoder:
    """Текстовый энкодер с поддержкой LaCLIP и Safetensors."""
    def __init__(self, model_name: str, device: torch.device, max_length: int = 128, pool: str = "mean"):
        self.model_name = model_name
        self.device = device
        self.pool = pool
        self.max_length = max_length

        # Загружаем конфиг через Safetensors
        cfg = AutoConfig.from_pretrained(model_name, use_safetensors=True)
        
        # Определяем тип архитектуры
        self._is_clip = any("CLIP" in arch for arch in getattr(cfg, "architectures", [])) or "clip" in model_name.lower()

        if self._is_clip:
            self.tok = CLIPTokenizer.from_pretrained(model_name)
            # Принудительно используем Safetensors для CLIP
            self.model = CLIPTextModel.from_pretrained(model_name, use_safetensors=True).to(device)
        else:
            self.tok = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name, use_safetensors=True).to(device)
            
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Обычное кодирование."""
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tok(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
            out = self.model(**inputs)
            
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                pooled = out.pooler_output
            else:
                pooled = pool_hidden(out.last_hidden_state, inputs["attention_mask"], mode=self.pool)
            
            all_embs.append(pooled.detach().cpu().numpy())
        return np.vstack(all_embs)
    
    @torch.no_grad()
    def encode_laclip(self, texts: List[str | None], texts_aug: List[List[str]]) -> np.ndarray:
        all_embs = []
        
        # CLIP имеет жёсткий лимит позиционных эмбеддингов
        clip_max_length = 77 if self._is_clip else self.max_length
        
        for i in range(len(texts)):
            orig_text = texts[i] if texts[i] is not None else ""
            variations = [orig_text]
            
            if i < len(texts_aug) and texts_aug[i]:
                variations.extend([str(a) for a in texts_aug[i] if a is not None])
            
            inputs = self.tok(
                variations, 
                padding=True, 
                truncation=True, 
                max_length=clip_max_length,  # <-- вот исправление
                return_tensors="pt"
            ).to(self.device)
            
            out = self.model(**inputs)
            
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                pooled = out.pooler_output
            else:
                pooled = pool_hidden(out.last_hidden_state, inputs["attention_mask"], mode=self.pool)
            
            mean_emb = pooled.mean(dim=0, keepdim=True)
            all_embs.append(mean_emb.detach().cpu().numpy())
            
        return np.vstack(all_embs)

    # @torch.no_grad()
    # def encode_laclip(self, texts: List[str | None], texts_aug: List[List[str]]) -> np.ndarray:
    #     """Метод LaCLIP: усреднение оригинала и аугментаций."""
    #     all_embs = []
    #     for i in range(len(texts)):
    #         orig_text = texts[i] if texts[i] is not None else ""
    #         variations = [orig_text]
            
    #         if i < len(texts_aug) and texts_aug[i]:
    #             variations.extend([str(a) for a in texts_aug[i] if a is not None])
            
    #         inputs = self.tok(
    #             variations, 
    #             padding=True, 
    #             truncation=True, 
    #             max_length=self.max_length, 
    #             return_tensors="pt"
    #         ).to(self.device)
            
    #         out = self.model(**inputs)
            
    #         if hasattr(out, "pooler_output") and out.pooler_output is not None:
    #             pooled = out.pooler_output
    #         else:
    #             pooled = pool_hidden(out.last_hidden_state, inputs["attention_mask"], mode=self.pool)
            
    #         # Усредняем все вариации одного текста
    #         mean_emb = pooled.mean(dim=0, keepdim=True)
    #         all_embs.append(mean_emb.detach().cpu().numpy())
            
    #     return np.vstack(all_embs)

class ImageEncoder:
    """Энкодер изображений CLIP с поддержкой Safetensors."""
    def __init__(self, model_name: str, device: torch.device):
        self.device = device
        self.model_name = model_name
        self.proc = CLIPProcessor.from_pretrained(model_name)
        # Принудительно используем Safetensors
        self.model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(device)
        self.model.eval()

    def _encode_batch(self, imgs: List) -> np.ndarray:
        inputs = self.proc(images=imgs, return_tensors="pt").to(self.device)
        out = self.model.get_image_features(pixel_values=inputs["pixel_values"])
        
        if isinstance(out, torch.Tensor):
            feats = out
        elif hasattr(out, "image_embeds"):
            feats = out.image_embeds
        else:
            feats = out.pooler_output
            
        return feats.detach().cpu().numpy()

    @torch.no_grad()
    def encode_paths(self, image_paths: List[str], batch_size: int = 32) -> np.ndarray:
        embs = []
        batch_imgs = []
        for p in image_paths:
            if p is None or not isinstance(p, str): continue
            try:
                img = Image.open(p).convert("RGB")
                batch_imgs.append(img)
            except Exception as e:
                print(f"Error loading {p}: {e}")
                continue

            if len(batch_imgs) == batch_size:
                embs.append(self._encode_batch(batch_imgs))
                batch_imgs = []

        if batch_imgs:
            embs.append(self._encode_batch(batch_imgs))

        return np.vstack(embs) if embs else np.array([])

class ViltEncoder:
    """Энкодер ViLT."""
    def __init__(self, model_name: str, device: torch.device):
        self.device = device
        self.model = ViltModel.from_pretrained(model_name, use_safetensors=True).to(device)
        self.proc = ViltProcessor.from_pretrained(model_name)
        self.model.eval()

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
                    b_imgs.append(Image.new("RGB", (224, 224), (0, 0, 0)))

            inputs = self.proc(images=b_imgs, text=b_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            out = self.model(**inputs)
            pooled = out.pooler_output
            embs.append(pooled.detach().cpu().numpy())
        return np.vstack(embs)