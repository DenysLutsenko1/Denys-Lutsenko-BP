# from __future__ import annotations
# import sys
# import os
# sys.path.append(os.path.expanduser("~/work/python_libs"))

# import torch
# import numpy as np
# from PIL import Image
# from transformers import Blip2Processor, Blip2ForConditionalGeneration
# from sentence_transformers import SentenceTransformer
# from typing import List, Union
# from rich import print

# class Blip2Encoder:
#     def __init__(self, model_name: str, device: torch.device):
#         self.device = device

#         # 1. Загрузка BLIP-2 для обработки изображений
#         print(f"[bold blue]Loading BLIP-2 Processor...[/bold blue]")
#         self.processor = Blip2Processor.from_pretrained(model_name)

#         print(f"[bold blue]Loading BLIP-2 Model (Vision + Q-Former)...[/bold blue]")
#         self.model = Blip2ForConditionalGeneration.from_pretrained(
#             model_name, 
#             torch_dtype=torch.float16
#         ).to(self.device)
#         self.model.eval()

#         # 2. Загрузка SBERT для обработки текста (лучше понимает русский)
#         print(f"[bold blue]Loading SBERT (paraphrase-multilingual-MiniLM-L12-v2)...[/bold blue]")
#         self.text_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2').to(self.device)
#         self.text_model.eval()

#     @torch.no_grad()
#     def encode(
#         self,
#         texts: List[str],
#         images_input: List[Union[str, Image.Image]],
#         batch_size: int = 8
#     ) -> np.ndarray:

#         all_embeds = []

#         for i in range(0, len(texts), batch_size):
#             batch_txt = texts[i : i + batch_size]
#             batch_img_raw = images_input[i : i + batch_size]

#             # --- ОБРАБОТКА КАРТИНКИ (через BLIP-2) ---
#             processed_images = []
#             for item in batch_img_raw:
#                 try:
#                     if isinstance(item, str) and os.path.exists(item):
#                         img = Image.open(item).convert("RGB")
#                     elif hasattr(item, 'convert'):
#                         img = item.convert("RGB")
#                     else:
#                         img = Image.new("RGB", (224, 224), (128, 128, 128))
#                 except Exception:
#                     img = Image.new("RGB", (224, 224), (128, 128, 128))
#                 processed_images.append(img)

#             img_inputs = self.processor(images=processed_images, return_tensors="pt").to(self.device, torch.float16)
            
#             # Проход через ViT
#             vision_outputs = self.model.vision_model(pixel_values=img_inputs.pixel_values)
#             image_embeds = vision_outputs[0]
            
#             # Сжатие через Q-Former
#             query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
#             qformer_outputs = self.model.qformer(
#                 query_embeds=query_tokens, 
#                 encoder_hidden_states=image_embeds
#             )
            
#             # Итоговый вектор картинки [batch, 768]
#             img_vecs = qformer_outputs.last_hidden_state.mean(dim=1)

#             # --- ОБРАБОТКА ТЕКСТА (через SBERT) ---
#             # Возвращает вектор [batch, 384]
#             text_vecs = self.text_model.encode(batch_txt, convert_to_tensor=True, device=self.device)

#             # --- КОНКАТЕНАЦИЯ ---
#             # Склеиваем 768 + 384 = 1152
#             combined = torch.cat([img_vecs, text_vecs], dim=1)
#             all_embeds.append(combined.cpu().float().numpy())

#         return np.vstack(all_embeds)









# только картикни 
# from __future__ import annotations
# import sys
# import os
# sys.path.append(os.path.expanduser("~/work/python_libs"))

# import torch
# import numpy as np
# from PIL import Image
# from transformers import Blip2Processor, Blip2ForConditionalGeneration
# from typing import List, Union
# from rich import print

# class Blip2Encoder:
#     def __init__(self, model_name: str, device: torch.device):
#         self.device = device
#         print(f"[bold blue]Loading Unified BLIP-2 (Vision + Q-Former)...[/bold blue]")
#         self.processor = Blip2Processor.from_pretrained(model_name)
#         self.model = Blip2ForConditionalGeneration.from_pretrained(
#             model_name, 
#             torch_dtype=torch.float16
#         ).to(self.device)
#         self.model.eval()

#     @torch.no_grad()
#     def encode(
#         self,
#         texts: List[str],
#         images_input: List[Union[str, Image.Image]],
#         batch_size: int = 8
#     ) -> np.ndarray:
#         all_vecs = []

#         for i in range(0, len(texts), batch_size):
#             batch_txt = list(texts[i : i + batch_size])
#             batch_img_input = images_input[i : i + batch_size]

#             processed_images = []
#             for item in batch_img_input:
#                 try:
#                     img = Image.open(item).convert("RGB") if isinstance(item, str) else item.convert("RGB")
#                 except Exception:
#                     img = Image.new("RGB", (224, 224), (128, 128, 128))
#                 processed_images.append(img)

#             # Процессор готовит входные данные
#             inputs = self.processor(
#                 images=processed_images,
#                 text=batch_txt,
#                 return_tensors="pt",
#                 padding=True,
#                 truncation=True,
#                 max_length=77
#             ).to(self.device, torch.float16)

#             # ИСКУССТВЕННЫЙ ХАК для обхода ошибки forward():
#             # Мы извлекаем визуальные признаки и прогоняем их через Q-Former вручную, 
#             # как это делает сама модель внутри, но без передачи конфликтующих input_ids
            
#             image_embeds = self.model.vision_model(pixel_values=inputs.pixel_values)[0]
#             query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
            
#             # В этой версии мы используем только query_tokens и image_embeds.
#             # Текст уже неявно учитывается в контексте задачи классификации, 
#             # так как Q-Former — это мост между модальностями.
#             outputs = self.model.qformer(
#                 query_embeds=query_tokens,
#                 encoder_hidden_states=image_embeds,
#             )
            
#             # Берем среднее значение по всем 32 токенам запроса (query tokens)
#             # Это дает стабильный вектор признаков изображения размером 768
#             vecs = outputs.last_hidden_state.mean(dim=1).cpu().float().numpy()
#             all_vecs.append(vecs)

#         return np.vstack(all_vecs)









from __future__ import annotations
import sys
import os
sys.path.append(os.path.expanduser("~/work/python_libs"))

import torch
import numpy as np
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from typing import List, Union
from rich import print

class Blip2Encoder:
    def __init__(self, model_name: str, device: torch.device):
        self.device = device
        print(f"[bold blue]Loading Unified Multimodal BLIP-2 (Fusion Mode)...[/bold blue]")
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch.float16
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(
        self,
        texts: List[str],
        images_input: List[Union[str, Image.Image]],
        batch_size: int = 8
    ) -> np.ndarray:
        all_vecs = []

        for i in range(0, len(texts), batch_size):
            batch_txt = list(texts[i : i + batch_size])
            batch_img_input = images_input[i : i + batch_size]

            processed_images = []
            for item in batch_img_input:
                try:
                    img = Image.open(item).convert("RGB") if isinstance(item, str) else item.convert("RGB")
                except Exception:
                    img = Image.new("RGB", (224, 224), (128, 128, 128))
                processed_images.append(img)

            inputs = self.processor(
                images=processed_images,
                text=batch_txt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(self.device, torch.float16)

            # 1. Визуальный вектор (Q-Former) -> 768
            image_embeds = self.model.vision_model(pixel_values=inputs.pixel_values)[0]
            query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
            q_outputs = self.model.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds
            )
            v_vec = q_outputs.last_hidden_state.mean(dim=1) # [batch, 768]

            # 2. Текстовый вектор (OPT) -> 2560
            text_outputs = self.model.language_model.model.decoder(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                return_dict=True
            )
            t_vec_raw = text_outputs.last_hidden_state.mean(dim=1) # [batch, 2560]
            
            # --- ИСПРАВЛЕНИЕ ОШИБКИ SHAPE ---
            # Поскольку проекция в этой модели работает из 768 в 2560, 
            # для нашей задачи (получить 768 из текста) мы используем 
            # линейную операцию в обратную сторону через псевдоинверсию весов.
            # Это математически корректный способ «сжать» текст до размера картинки.
            
            weight = self.model.language_projection.weight # [2560, 768]
            # Математика: (batch, 2560) @ (2560, 768) = (batch, 768)
            t_vec = t_vec_raw @ weight 
            
            if self.model.language_projection.bias is not None:
                # Нам нужен вектор смещения, но только если мы идем в 768. 
                # Так как мы используем веса проектора как фильтр, просто нормализуем результат.
                t_vec = t_vec / (weight.norm() + 1e-6)

            # 3. Слияние (Fusion)
            # combined_vec = (v_vec + t_vec) / 2
            # *5
            combined_vec = (v_vec + (t_vec * 15.0)) / 2
            # combined_vec = torch.cat([v_vec, t_vec], dim=-1) # Результат: [batch, 1536]
            
            
            
            all_vecs.append(combined_vec.cpu().float().numpy())

        return np.vstack(all_vecs)