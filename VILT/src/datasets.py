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

def texts_images_labels(data: List[Dict]):
    valid_texts = []
    valid_images = []
    valid_labels = []
    
    # Находим корень проекта (bp_latex)
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_file_dir, "..", ".."))
    
    # Печатаем для проверки в консоль
    print(f"DEBUG: Project root: {project_root}")

    for d in data:
        text = d.get("text")
        img_path = d.get("image_path") or d.get("image")
        label = d.get("label", -1)
        
        if text and isinstance(text, str) and text.strip() and img_path:
            # Склеиваем путь напрямую от КОРНЯ проекта
            # Если в JSON написано "images/real/1.jpg", 
            # то путь будет "C:/.../bp_latex/images/real/1.jpg"
            full_path = os.path.normpath(os.path.join(project_root, img_path))
            
            if os.path.exists(full_path):
                cleaned_text = " ".join(text.split())
                valid_texts.append(cleaned_text)
                valid_images.append(full_path)
                valid_labels.append(label)
            # else:
            #    print(f"DEBUG: Not found: {full_path}") # Раскомментируй, если опять будет 0
    
    print(f"Loaded {len(valid_texts)} valid samples from dataset.")
    return valid_texts, valid_images, valid_labels