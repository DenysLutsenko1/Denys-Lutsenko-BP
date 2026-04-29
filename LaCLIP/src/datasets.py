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
    texts = [d["text"] for d in data]
    labels = [d.get("label", -1) for d in data]
    return texts, labels

def texts_images_labels(data):
    # Находим корень проекта (bp_latex)
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_file_dir, "..", ".."))
    
    texts = []
    images = []
    labels = []
    
    for d in data:
        img_path = d.get("image_path") or d.get("image")
        if img_path:
            full_path = os.path.normpath(os.path.join(project_root, img_path))
            if os.path.exists(full_path):
                texts.append(d.get("text"))
                images.append(full_path)
                labels.append(d.get("label", -1))
    return texts, images, labels

def texts_aug_images_labels(data):
    """Специальная функция для LaCLIP с поддержкой аугментаций."""
    # Находим корень проекта (bp_latex)
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_file_dir, "..", ".."))
    
    texts = []
    texts_aug = []
    images = []
    labels = []
    
    for d in data:
        img_path = d.get("image_path") or d.get("image")
        
        if img_path:
            # Склеиваем путь: BP_LATEX + images/gen/1.jpg
            full_path = os.path.normpath(os.path.join(project_root, img_path))
            
            # Если картинка существует, берем и её, и все тексты
            if os.path.exists(full_path):
                texts.append(d.get("text"))
                texts_aug.append(d.get("text_aug", [])) 
                images.append(full_path)
                labels.append(d.get("label", -1))
                
    return texts, texts_aug, images, labels