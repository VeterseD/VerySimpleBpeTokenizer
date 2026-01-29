"""
Обучение BPE токенайзера на Rust (быстрая версия)
Использует Rust реализацию для ускорения в 5-10x
"""

import json
import os
import sys

try:
    from bpe_encode_rust import PyBpeTrainer
except ImportError:
    print("❌ Rust модуль не найден!")
    print("Сначала скомпилируй Rust код:")
    print("  cd bpe/encode_rs")
    print("  cargo build --release --features python")
    print("  cd ../..")
    print("  copy bpe\\encode_rs\\target\\release\\bpe_encode_rust.dll bpe_encode_rust.pyd")
    sys.exit(1)


def train_tokenizer_from_json_rust(
    json_file: str,
    vocab_size: int = 20000,
    output_path: str = 'tokenizer',
    num_threads: int = 4
):
    """
    Обучает токенайзер на данных из JSON используя Rust.
    
    Args:
        json_file: Путь к JSON файлу с данными
        vocab_size: Размер словаря
        output_path: Префикс для сохранения (без расширения)
        num_threads: Количество потоков для параллельной обработки
    """
    print(f"Загрузка данных из {json_file}...")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Создаем временный текстовый файл с разделителем
    temp_file = 'temp_training_data.txt'
    special_token = "<|endoftext|>"
    
    print(f"Подготовка данных для обучения...")
    with open(temp_file, 'w', encoding='utf-8') as f:
        for item in data:
            text_parts = []
            
            # Title
            if item.get('title'):
                text_parts.append(f"Title: {item['title']}")
            
            # Content
            if item.get('content') and len(item['content']) > 50:
                text_parts.append(item['content'])
            
            # Quotes с персонажами
            if item.get('quotes'):
                for quote in item['quotes']:
                    character = quote.get('character', 'Unknown')
                    quote_text = quote.get('quote', '')
                    if quote_text:
                        text_parts.append(f"{character}: \"{quote_text}\"")
            
            if text_parts:
                f.write('\n\n'.join(text_parts))
                f.write(f"\n{special_token}\n")
    
    print(f"✓ Данные подготовлены")
    
    # Обучаем токенайзер на Rust
    print(f"\n🚀 Запуск обучения на Rust (vocab_size={vocab_size}, threads={num_threads})...")
    
    trainer = PyBpeTrainer(None)  # None = default regex pattern
    
    trainer.train(
        input_file_path=temp_file,
        vocab_size=vocab_size,
        special_tokens=[special_token],
        boundary_split_token=special_token,
        num_threads=num_threads,
        verbose=True
    )
    
    # Регистрируем special token
    trainer.register_special_tokens({special_token: vocab_size})
    
    # Сохраняем
    trainer.save(output_path)
    
    # Удаляем временный файл
    os.remove(temp_file)
    print(f"✓ Временный файл удален")
    
    print(f"\n✓ Обучение завершено!")
    print(f"Файлы: {output_path}.model, {output_path}.vocab")


if __name__ == "__main__":
    # Обучаем токенайзер на твоих данных
    train_tokenizer_from_json_rust(
        json_file='crawled_data_clean.json',
        vocab_size=20000,
        output_path='tokenizer',
        num_threads=4  # Используем 4 потока
    )
