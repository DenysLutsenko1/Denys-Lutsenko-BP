

# что бы оно корреткно работало то нужно убрать sys.path.append(os.path.expanduser("~/work/python_libs"))

import os
import torch
import joblib
import io
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

# ВАЖНО: Никаких sys.path.append с путями /home/jovyan/...
# Локальные импорты:
try:
    from src.encoders import TextEncoder, ImageEncoder
    from src.models import TorchMLP
except ImportError:
    # Если запуск идет напрямую из папки src
    from encoders import TextEncoder, ImageEncoder
    from models import TorchMLP

app = Flask(__name__)
CORS(app) 

# Правильные пути для Windows
# Находим папку LaCLIP (на уровень выше, чем src)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE_PATH = os.path.join(BASE_DIR, "outputs", "multimodal.joblib")

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

# Глобальные переменные для энкодеров и модели
txt_enc = None
img_enc = None
clf = None

def load_model():
    global txt_enc, img_enc, clf
    print(f"[*] Попытка загрузки бандла: {BUNDLE_PATH}")
    
    if not os.path.exists(BUNDLE_PATH):
        raise FileNotFoundError(f"Файл модели не найден по пути: {BUNDLE_PATH}")

    bundle = joblib.load(BUNDLE_PATH)
    
    # Инициализация энкодеров
    txt_enc = TextEncoder(
        bundle["text"]["encoder_name"], 
        DEVICE, 
        max_length=bundle["text"].get("max_length", 256)
    )
    img_enc = ImageEncoder(bundle["image"]["encoder_name"], DEVICE)
    
    # Загрузка пайплайна
    clf = bundle["pipeline"]
    
    # Перевод в режим предсказания (выключаем Dropout)
    if hasattr(clf.named_steps['clf'], 'model'):
        clf.named_steps['clf'].model.eval()
        clf.named_steps['clf'].model.to(DEVICE)
    
    print("[+] Все компоненты загружены!")

# Загружаем сразу при старте
try:
    load_model()
except Exception as e:
    print(f"[-] КРИТИЧЕСКАЯ ОШИБКА ЗАГРУЗКИ: {e}")

def get_similarity(tvec, ivec):
    t_norm = np.linalg.norm(tvec)
    i_norm = np.linalg.norm(ivec)
    if t_norm > 1e-9 and i_norm > 1e-9:
        return float(np.dot(tvec / t_norm, ivec / i_norm))
    return 0.0

@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.form.get('text', '')
        file = request.files.get('image')

        if not file or not text:
            return jsonify({"error": "Missing text or image"}), 400

        # Кодируем
        t_emb = txt_enc.encode([text])[0]
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        i_emb = img_enc._encode_batch([image])[0]

        # Фичи
        sim = get_similarity(t_emb, i_emb)
        fused = np.concatenate([t_emb, i_emb, [sim]]).reshape(1, -1)

        # Вероятность
        probs = clf.predict_proba(fused)[0]
        # Обычно индекс 1 — это класс "AI/GenAI"
        prob_val = float(probs[1])

        fused_tensor = torch.as_tensor(fused, dtype=torch.float32).to(DEVICE)

        ai_score = float(probs[1]) * 100 
        

        return jsonify({
            "ai_probability": f"{ai_score:.2f}%",
            "is_ai": ai_score > 50,
            "status": "success"
        })

    except Exception as e:
        # ПЕЧАТАЕМ ОШИБКУ В КОНСОЛЬ, чтобы ты её увидел
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)


