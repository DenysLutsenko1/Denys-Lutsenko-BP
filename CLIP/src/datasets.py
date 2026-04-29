from __future__ import annotations

import sys
import os
sys.path.append(os.path.expanduser("~/work/python_libs"))





import json
import os
from typing import List, Dict

def load_jsonl(path: str) -> List[Dict]:
    """Загрузка данных из JSONL."""
    data = []
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        print(f"Error: File {abs_path} not found!")
        return []

    with open(abs_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def texts_and_labels(data):
    """Тексты и метки без обработки путей картинок."""
    texts = [d["text"] for d in data]
    labels = [d.get("label", -1) for d in data]
    return texts, labels

# def texts_images_labels(data):
#     """Тексты, полные пути к картинкам и метки."""
#     # Определяем корень проекта (поднимаемся из CLIP/src на два уровня вверх)
#     current_file_dir = os.path.dirname(os.path.abspath(__file__))
#     project_root = os.path.abspath(os.path.join(current_file_dir, "..", ".."))
    
#     texts = []
#     images = []
#     labels = []
    
#     for d in data:
#         text = d.get("text")
#         img_path = d.get("image_path") or d.get("image")
#         label = d.get("label", -1)
        
#         if img_path:
#             # Склеиваем путь от корня (например, C:/.../BP_LATEX/ + images/gen/1.jpg)
#             full_path = os.path.normpath(os.path.join(project_root, img_path))
            
#             # ПРОВЕРКА: добавляем только если файл РЕАЛЬНО существует
#             if os.path.exists(full_path):
#                 texts.append(text)
#                 images.append(full_path)
#                 labels.append(label)
#             else:
#                 # Если файла нет (как empty.jpg), просто пропускаем эту строку,
#                 # чтобы CLIP не вылетал с FileNotFoundError
#                 continue
        
#     return texts, images, labels
def texts_images_labels(data):
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_file_dir, "..", ".."))
    
    texts, images, labels = [], [], []
    
    print("\n--- ПРОВЕРКА ПЕРВЫХ 10 ПУТЕЙ ---")
    found_count = 0
    
    for d in data:
        text = d.get("text")
        img_path = d.get("image_path") or d.get("image")
        label = d.get("label", -1)
        
        if img_path:
            full_path = os.path.normpath(os.path.join(project_root, img_path))
            
            if os.path.exists(full_path):
                texts.append(text)
                images.append(full_path)
                labels.append(label)
                
                # Печатаем первые 10 успешных находок
                if found_count < 10:
                    print(f"[{found_count}] FOUND: {full_path}")
                    found_count += 1
            else:
                # Если вдруг что-то не так, это тоже увидим
                print(f"[ERROR] NOT FOUND: {full_path}")
        
    print(f"-------------------------------\n")
    return texts, images, labels