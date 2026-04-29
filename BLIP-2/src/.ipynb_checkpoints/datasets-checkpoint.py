from __future__ import annotations
import sys
import os
sys.path.append(os.path.expanduser("~/work/python_libs"))

import json
import random
from typing import List, Dict, Tuple


def load_jsonl(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл не найден: {path}")

    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    random.seed(42)
    random.shuffle(data)
    return data


def texts_images_labels(
    data: List[Dict],
    project_root: str | None = None
) -> Tuple[List[str], List[str], List[int]]:

    if project_root is None:
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_file_dir, "..", ".."))

    valid_texts:  List[str] = []
    valid_images: List[str] = []
    valid_labels: List[int] = []

    skipped_no_label = 0
    skipped_no_text  = 0
    skipped_no_image = 0

    for d in data:
        text      = d.get("text")
        img_rel   = d.get("image_path")
        label     = d.get("label")

        if label is None:
            skipped_no_label += 1
            continue

        if not (text and isinstance(text, str) and text.strip()):
            skipped_no_text += 1
            continue

        if not img_rel:
            skipped_no_image += 1
            continue

        full_path = os.path.normpath(os.path.join(project_root, img_rel))
        if not os.path.exists(full_path):
            skipped_no_image += 1
            continue

        valid_texts.append(text.strip())
        valid_images.append(full_path)
        valid_labels.append(int(label))

    print(f"--- Dataset Report ---")
    print(f"Project root     : {project_root}")
    print(f"Total in JSONL   : {len(data)}")
    print(f"Valid pairs      : {len(valid_labels)}")
    print(f"Skipped no label : {skipped_no_label}")
    print(f"Skipped no text  : {skipped_no_text}")
    print(f"Skipped no image : {skipped_no_image}")
    print(f"----------------------")

    if len(valid_labels) == 0:
        raise RuntimeError("No valid samples found! Check JSONL and image paths.")

    return valid_texts, valid_images, valid_labels