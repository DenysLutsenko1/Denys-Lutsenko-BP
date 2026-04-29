# Project Commands

eval.py
python -m src.eval --config config.yaml --ckpt outputs/multimodal_clf.joblib

infer_multi.py
python -m src.infer_multi --ckpt outputs/multimodal_clf.joblib --text "qwe" --image images/real/6.jpg

python -m src.infer_multi.py --ckpt outputs/vilt_model.joblib \
  --text "Кот" "Собака" \
  --image "cat.jpg" "dog.jpg"

train_multi.py
python -m src.train_multi --config config.yaml