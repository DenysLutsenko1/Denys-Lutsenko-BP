
import sys
import os
import torch
import joblib
import io
import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

# Импортируем твои классы
try:
    from src.encoders import TextEncoder, ImageEncoder
except ImportError:
    # Если запуск идет напрямую из папки src
    from encoders import TextEncoder, ImageEncoder
# from encoders import TextEncoder, ImageEncoder

app = Flask(__name__)
CORS(app) 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE_PATH = os.path.join(BASE_DIR, "outputs", "multimodal_clf.joblib")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[*] Загрузка модели из: {BUNDLE_PATH}")

try:
    # Загружаем новый бандл
    bundle = joblib.load(BUNDLE_PATH)
    clf = bundle["pipeline"]
    cfg = bundle.get("config") # Используем .get чтобы не падать если ключа нет
    
    if cfg is None:
        raise ValueError("В файле модели отсутствует ключ 'config'. Переобучите модель новым скриптом.")

    # Инициализация энкодеров по новому конфигу
    txt_enc = TextEncoder(
        cfg["encoder"]["name"], 
        DEVICE, 
        max_length=cfg["encoder"]["max_length"]
    )
    img_enc = ImageEncoder(cfg["image_encoder"]["name"], DEVICE)
    
    print("[+] Модель и энкодеры успешно загружены!")
except Exception as e:
    print(f"[-] Ошибка инициализации: {e}")
    # Чтобы приложение не падало сразу, но выводило ошибку в консоль

def fuse_single(tvec, ivec, use_similarity=True):
    # Косинусное сходство (логика из train_multi)
    sim = 0.0
    if use_similarity:
        tv_norm = np.linalg.norm(tvec)
        iv_norm = np.linalg.norm(ivec)
        if tv_norm > 1e-9 and iv_norm > 1e-9:
            sim = float(np.dot(tvec / tv_norm, ivec / iv_norm))
    
    # Конкатенация (512 + 512 + 1 = 1025)
    return np.concatenate([tvec, ivec, np.array([sim], dtype=np.float32)], axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.form.get('text', '')
        file = request.files.get('image')

        if not file or not text:
            return jsonify({"error": "Нужны текст и картинка"}), 400

        # Кодируем
        t_emb = txt_enc.encode([text], batch_size=1)[0]
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        i_emb = img_enc._encode_batch([img])[0]

        # Фузия (1025 признаков)
        use_sim = cfg.get("multimodal", {}).get("use_similarity", True)
        fused_vector = fuse_single(t_emb, i_emb, use_similarity=use_sim)
        X = fused_vector.reshape(1, -1)

        # Предсказание
        proba = clf.predict_proba(X)[0]
        ai_score = float(proba[1]) * 100 
        

        return jsonify({
            "ai_probability": f"{ai_score:.2f}%",
            "is_ai": ai_score > 50,
            "status": "success"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Используем порт 5000 как в твоем логе
    app.run(host='0.0.0.0', port=5000)