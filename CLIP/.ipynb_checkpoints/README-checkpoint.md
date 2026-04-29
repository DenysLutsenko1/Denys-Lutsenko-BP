# Project Commands

Train Text-Only Classifier
python -m src.train --config config.yaml

Train Multimodal Classifier (Text + Image)
python -m src.train_multi --config config.yaml

Evaluate Text-Only Classifier
python -m src.eval --config config.yaml --ckpt outputs/text_clf.joblib

Inference (Text Only)
python -m src.infer --ckpt outputs/text_clf.joblib --text "Hello world"

Check Text and Image:   
python -m src.infer_multi --ckpt outputs/multimodal_clf.joblib --text "123" --image images/real/1.jpg

Check Text Only (via multimodal script):
python -m src.infer_multi --ckpt outputs/multimodal_clf.joblib --text "Text verification"





\\\\
проверить работает ли текст просто или просто картинка спросить в понедельник
добаивть что то про аудио так что бы баларе понрвилось суббота
\\\\\\\


\\\\\
блип 
тренировка
визуальная часть 
общий путь
\\\\

\\\\\
посмотреть датасет балары пятница - пятница 
сделать аугментированный датасет 
\\\\\\


\\\\\\\
поиграться с конфигами для размера текстов разные модели поделать 

\\\\\\
расширить датасет 