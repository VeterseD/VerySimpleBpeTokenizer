"""
Тестирование обученного токенайзера
"""

import sys
sys.path.insert(0, '.')

from bpe_encode_rust import PyBpeEncode

# Загружаем токенайзер
encoder = PyBpeEncode()
encoder.load("tokenizer.model")

# Тестовые тексты
test_texts = [
    "Hello world!",
    "Привет мир!",
    "The quick brown fox jumps over the lazy dog.",
    "Это тестовый текст на русском языке.",
    "Title: Test\n\nContent: This is a test.",
]

print("=" * 60)
print("Тест токенайзера")
print("=" * 60)

for i, text in enumerate(test_texts, 1):
    print(f"\n{i}. Текст: {text[:50]}...")
    
    # Кодируем
    ids = encoder.encode(text, "none")
    print(f"   Токенов: {len(ids)}")
    print(f"   IDs: {ids[:20]}{'...' if len(ids) > 20 else ''}")
    
    # Декодируем
    decoded = encoder.decode(ids)
    
    # Проверяем roundtrip
    if decoded == text:
        print(f"   ✓ Roundtrip OK")
    else:
        print(f"   ✗ Roundtrip FAIL")
        print(f"   Original:  {repr(text[:100])}")
        print(f"   Decoded:   {repr(decoded[:100])}")
    
    # Compression ratio
    compression = len(text) / len(ids) if len(ids) > 0 else 0
    print(f"   Compression: {compression:.2f}x")

print("\n" + "=" * 60)
print("Проверка vocab size")
print("=" * 60)

# Проверяем сколько токенов в vocab
with open("tokenizer.vocab", "r", encoding="utf-8") as f:
    lines = f.readlines()
    vocab_size = len(lines)
    print(f"Vocab size: {vocab_size}")
    print(f"Ожидалось: 20000")
    print(f"Разница: {20000 - vocab_size}")
    
    # Показываем последние токены
    print(f"\nПоследние 10 токенов:")
    for line in lines[-10:]:
        print(f"  {line.strip()}")
