import random

# Укажите имя вашего файла
input_filename = 'trainwithnew.jsonl'
output_filename = 'trainwithnewshuffled.jsonl'

# Читаем все строки из файла
with open(input_filename, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Перемешиваем список строк случайным образом
random.shuffle(lines)

# Записываем перемешанные строки в новый файл
with open(output_filename, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f"Файл успешно перемешан! Результат в: {output_filename}")