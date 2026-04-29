import sys
import os

import torch
import joblib
import io
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

# Импортируем энкодер и утилиту для фиксации сида
try:
    from src.encoders import ViltEncoder
    from src.utils import set_seed
except ImportError:
    from encoders import ViltEncoder
    from utils import set_seed

# ФИКСАЦИЯ СИДА: Гарантирует, что случайные веса позиционных эмбеддингов (41-512) 
# совпадут с теми, что были при обучении.
set_seed(42)

app = Flask(__name__)
CORS(app) 

# Определяем устройство. Если обучение шло на cuda:1, рекомендуется использовать тот же индекс.
DEVICE = torch.device("cpu")

# Путь к модели
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE_PATH = os.path.join(BASE_DIR, "outputs", "multimodal_clf.joblib")

vilt_enc = None
clf = None

print(f"[*] Loading ViLT Bundle from: {BUNDLE_PATH}")
try:
    # Загружаем бандл, созданный в train_multi.py
    bundle = joblib.load(BUNDLE_PATH)
    
    model_name = bundle.get("model_name", "dandelin/vilt-b32-mlm")
    # Извлекаем сохраненный лимит токенов (например, 512)
    max_len = bundle.get("max_length", 40)
    
    print(f"[i] Model context length detected: {max_len}")
    
    # Инициализация энкодера с расширенным контекстом
    vilt_enc = ViltEncoder(model_name, DEVICE, max_length=max_len)
    clf = bundle["pipeline"]
    
    print(f"[+] Server is ready! Model: {model_name} | Device: {DEVICE}")
except Exception as e:
    print(f"[-] Loading error: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    if vilt_enc is None or clf is None:
        return jsonify({"error": "Model not loaded"}), 500
        
    try:
        text = request.form.get('text', '')
        file = request.files.get('image')

        if not file or not text:
            return jsonify({"error": "Text and image are required"}), 400

        # Очистка текста аналогично этапу загрузки данных
        clean_text = " ".join(text.split())

        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # ВАЖНО: В новой версии энкодера нужно явно передать dropout_prob=0.0.
        # Это отключает Text Dropout для получения точного предсказания.
        embedding = vilt_enc.encode(
            texts=[clean_text], 
            images_input=[img], 
            batch_size=1, 
            dropout_prob=0.0
        )

        # Предсказание вероятности с помощью MLP/LogisticRegression из бандла
        proba = clf.predict_proba(embedding)[0]
        ai_score = float(proba[1]) * 100 

        return jsonify({
            "ai_probability": f"{ai_score:.2f}%",
            "is_ai": ai_score > 50,
            "status": "success"
        })

    except Exception as e:
        print(f"[-] Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # В продакшене лучше использовать gunicorn, но для тестов оставляем так
    app.run(host='0.0.0.0', port=5000, debug=False)