# Project Commands

Train Multimodal Classifier (Text + Image)
python -m src.train_multi --config config.yaml

Check Text and Image:   
python -m src.infer_multi --ckpt outputs/multimodal.joblib --text "qwe" --image images/real/1.jpg

eval.py
python -m src.eval --config config.yaml --ckpt outputs/multimodal.joblib
