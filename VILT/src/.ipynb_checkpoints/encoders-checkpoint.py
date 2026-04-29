# from __future__ import annotations

# import sys
# import os
# sys.path.append(os.path.expanduser("~/work/python_libs"))

# import torch
# import numpy as np
# from PIL import Image
# from transformers import ViltProcessor, ViltModel, ViltConfig
# from typing import List, Union

# class ViltEncoder:
#     def __init__(self, model_name: str, device: torch.device, max_length: int = 512):
#         """
#         Инициализация ViLT с расширенным контекстом текста.
#         """
#         self.device = device
#         self.max_length = max_length
        
#         # 1. Загружаем конфигурацию и меняем лимит позиционных эмбеддингов
#         # По умолчанию у ViLT это 40, мы ставим 512 (или из конфига)
#         config = ViltConfig.from_pretrained(model_name)
#         config.max_position_embeddings = max_length 
        
#         # 2. Процессор для обработки изображений и текста
#         self.processor = ViltProcessor.from_pretrained(model_name)
        
#         # 3. Загрузка весов с новым конфигом
#         # ignore_mismatched_sizes=True позволяет инициализировать новые позиции (41-512)
#         # случайными весами, которые затем настроятся при обучении
#         self.model = ViltModel.from_pretrained(
#             model_name, 
#             config=config, 
#             ignore_mismatched_sizes=True,
#             use_safetensors=True
#         ).to(device)
        
#         self.model.eval()

#     @torch.no_grad()
#     def encode(self, texts: List[str], images_input: List[Union[str, Image.Image]], batch_size: int = 16) -> np.ndarray:
#         all_embeds = []
        
#         for i in range(0, len(texts), batch_size):
#             batch_txt = texts[i : i + batch_size]
#             batch_img_input = images_input[i : i + batch_size]
            
#             images = []
#             for item in batch_img_input:
#                 if isinstance(item, str) and item:
#                     try:
#                         img = Image.open(item).convert("RGB")
#                     except Exception:
#                         img = Image.new("RGB", (224, 224), (0, 0, 0))
#                 elif isinstance(item, Image.Image):
#                     img = item.convert("RGB")
#                 else:
#                     img = Image.new("RGB", (224, 224), (0, 0, 0))
#                 images.append(img)

#             # 4. Обработка через ViLT Processor с жестким лимитом
#             # padding="max_length" и truncation=True гарантируют стабильность тензоров
#             inputs = self.processor(
#                 images, 
#                 batch_txt, 
#                 return_tensors="pt", 
#                 padding="max_length", 
#                 truncation=True, 
#                 max_length=self.max_length
#             ).to(self.device)
            
#             outputs = self.model(**inputs)
            
#             # pooler_output (мультимодальный вектор 768 признаков)
#             all_embeds.append(outputs.pooler_output.cpu().numpy())
            
#         return np.vstack(all_embeds)


from __future__ import annotations
import sys
import os
sys.path.append(os.path.expanduser("~/work/python_libs"))

import torch
import numpy as np
from PIL import Image
from transformers import ViltProcessor, ViltModel, ViltConfig
from typing import List, Union

class ViltEncoder:
    def __init__(self, model_name: str, device: torch.device, max_length: int = 512):
        self.device = device
        self.max_length = max_length
        
        # Загружаем конфигурацию и расширяем лимит позиционных эмбеддингов
        config = ViltConfig.from_pretrained(model_name)
        config.max_position_embeddings = max_length 
        
        self.processor = ViltProcessor.from_pretrained(model_name)
        
        # Инициализируем модель с поддержкой изменения размеров (41 -> 512)
        self.model = ViltModel.from_pretrained(
            model_name, 
            config=config, 
            ignore_mismatched_sizes=True,
            use_safetensors=True
        ).to(device)
        
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: List[str], images_input: List[Union[str, Image.Image]], 
               batch_size: int = 16, dropout_prob: float = 0.0) -> np.ndarray:
        """
        Кодирование пар текст-изображение.
        dropout_prob: вероятность зануления текста для обучения визуальной ветки.
        """
        all_embeds = []
        
        for i in range(0, len(texts), batch_size):
            # Создаем копию батча текста для манипуляций
            batch_txt = list(texts[i : i + batch_size])
            batch_img_input = images_input[i : i + batch_size]
            
            # --- TEXT DROPOUT LOGIC ---
            # Если активен дропаут, заменяем часть текстов на пустые строки
            if dropout_prob > 0:
                for idx in range(len(batch_txt)):
                    if np.random.random() < dropout_prob:
                        batch_txt[idx] = "" 

            # Подготовка изображений
            images = []
            for item in batch_img_input:
                if isinstance(item, str) and item:
                    try:
                        img = Image.open(item).convert("RGB")
                    except Exception:
                        img = Image.new("RGB", (224, 224), (0, 0, 0))
                elif isinstance(item, Image.Image):
                    img = item.convert("RGB")
                else:
                    img = Image.new("RGB", (224, 224), (0, 0, 0))
                images.append(img)

            # Обработка через ViLT Processor
            inputs = self.processor(
                images, 
                batch_txt, 
                return_tensors="pt", 
                padding="max_length", 
                truncation=True, 
                max_length=self.max_length
            ).to(self.device)
            
            # Извлекаем мультимодальный вектор (pooler_output)
            outputs = self.model(**inputs)
            all_embeds.append(outputs.pooler_output.cpu().numpy())
            
        return np.vstack(all_embeds)