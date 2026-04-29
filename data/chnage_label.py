import json

input_file = 'train.jsonl'
output_file = 'train_updated.jsonl'

processed_count = 0
error_count = 0

with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8') as outfile:
    
    for i, line in enumerate(infile, 1):
        line = line.strip() # Убираем лишние пробелы и пустые строки
        if not line:
            continue
            
        try:
            data = json.loads(line)
            
            if 'gen' in data.get('image_path', ''):
                data['label'] = 1
                processed_count += 1
            
            json.dump(data, outfile, ensure_ascii=False)
            outfile.write('\n')
            
        except json.JSONDecodeError as e:
            print(f"Ошибка в строке {i}: {e}")
            error_count += 1

print(f"---")
print(f"Успешно изменено: {processed_count}")
print(f"Ошибок в формате: {error_count}")