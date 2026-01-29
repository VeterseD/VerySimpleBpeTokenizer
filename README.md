# Оптимизированный BPE Tokenizer Training (Rust)

Быстрая реализация обучения BPE токенайзера на Rust с Python bindings.

## Особенности

- **Flat corpus с linked list** - минимум аллокаций
- **HashSet индекс позиций пар** - O(1) операции
- **Priority Queue (BinaryHeap)** - O(log N) поиск максимума
- **Инкрементальное обновление** - обновляем только затронутые пары
- **Параллельная пре-токенизация** - multithread обработка

## Структура файлов

```
bpe/
└── encode_rs/
    ├── Cargo.toml              # Rust dependencies
    ├── src/
    │   ├── lib.rs              # Python bindings
    │   ├── bpe_encode.rs       # Encoding (существующий)
    │   └── bpe_train.rs        # Training (новый)
    └── target/                 # Скомпилированные файлы

train_tokenizer_rust.py         # Скрипт обучения
test_tokenizer.py               # Тестирование
```

## Установка

### 1. Установить Rust

```bash
# Windows
https://rustup.rs/

# Linux/Mac
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### 2. Скомпилировать Rust модуль

```bash
cd bpe/encode_rs
cargo build --release --features python
cd ../..
```

### 3. Скопировать скомпилированный модуль

**Windows:**
```cmd
copy bpe\encode_rs\target\release\bpe_encode_rust.dll bpe_encode_rust.pyd
```

**Linux:**
```bash
cp bpe/encode_rs/target/release/libbpe_encode_rust.so bpe_encode_rust.so
```

**Mac:**
```bash
cp bpe/encode_rs/target/release/libbpe_encode_rust.dylib bpe_encode_rust.so
```

## Использование

### Обучение токенайзера

```python
from train_tokenizer_rust import train_tokenizer_from_json_rust

# Обучить на JSON данных
train_tokenizer_from_json_rust(
    json_file='your_data.json',
    vocab_size=20000,
    output_path='tokenizer',
    num_threads=4
)
```

### Формат входных данных

JSON файл с массивом объектов:

```json
[
  {
    "title": "Заголовок",
    "content": "Основной текст...",
    "quotes": [
      {
        "character": "Персонаж",
        "quote": "Цитата..."
      }
    ]
  }
]
```

### Тестирование

```python
python test_tokenizer.py
```

## Производительность

**Сравнение с Python версией:**

| Реализация | Время | Ускорение |
|------------|-------|-----------|
| Python (наивная) | 2.4 часа | 1x |
| Python (оптимизированная) | 41 минута | 3.5x |
| Rust (эта версия) | ~5-10 минут | 15-30x |

**Тестовые данные:** 32k текстов, 21M токенов, vocab_size=20000

## Технические детали

### Оптимизации

1. **Flat Corpus**
   - Один `Vec<u32>` для всех токенов
   - Linked list через индексы (`next`, `prev`)
   - Маркер `DELETED` вместо реального удаления

2. **HashSet для позиций**
   - `HashMap<(u32, u32), HashSet<usize>>`
   - O(1) добавление/удаление позиций
   - Быстрый поиск пар

3. **Priority Queue**
   - `BinaryHeap<PairFreq>` для частот
   - O(log N) извлечение максимума
   - Lazy deletion для stale записей

4. **Инкрементальные обновления**
   - После merge обновляем только пары с новым токеном
   - Добавляем в heap только измененные пары
   - Проверка актуальности при извлечении

### Сложность

- **Пре-токенизация:** O(T) где T = размер текста
- **Построение индекса:** O(N) где N = количество токенов
- **Один merge:** O(K * log M) где K = вхождения пары, M = уникальных пар
- **Все merges:** O(V * K_avg * log M) где V = vocab_size

## Troubleshooting

### Ошибка компиляции

```
error: look-around is not supported
```

**Решение:** Используется обычный `regex` вместо `fancy-regex`, lookahead не поддерживается.

### Модуль не найден

```
ImportError: No module named 'bpe_encode_rust'
```

**Решение:** Проверь что `.pyd` (Windows) или `.so` (Linux/Mac) файл в текущей директории.

### Останавливается на N merges

```
No more pairs to merge after 1115 merges
```

**Решение:** Убедись что используешь последнюю версию с инкрементальным обновлением heap.

## Лицензия

MIT

