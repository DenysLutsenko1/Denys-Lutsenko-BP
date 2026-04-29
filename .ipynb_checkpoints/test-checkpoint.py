import sys
import os

# Указываем путь к твоим библиотекам
sys.path.append(os.path.expanduser("~/work/python_libs"))

import torch

print("--- Финальная проверка мощности ---")
if torch.cuda.is_available():
    # Мы принудительно выберем вторую карту (индекс 0, так как мы скроем остальные)
    device = torch.device("cuda")
    
    print(f"✅ Успех! Видеокарта определена.")
    print(f"Имя карты: {torch.cuda.get_device_name(0)}")
    print(f"Свободно памяти на карте: {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GB")

    # Сделаем тяжелое вычисление для теста
    a = torch.randn(10000, 10000).to(device)
    b = torch.randn(10000, 10000).to(device)
    print("Начинаю умножение огромных матриц на GPU...")
    c = torch.matmul(a, b)
    print("✅ Вычисление завершено успешно!")
else:
    print("❌ GPU не найден.")