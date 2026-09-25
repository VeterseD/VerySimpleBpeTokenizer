# azchess

Шахматный движок в стиле AlphaZero на PyTorch: ResNet (policy + value), PUCT MCTS, self-play, обучение на CUDA и UCI для игры в GUI.

## Установка

PyTorch с CUDA уже стоит в твоём venv, нужно только найти его и доставить `chess` и `numpy`.

```bash
# Linux: найти venv с torch
find ~ -type d -path "*site-packages/torch" 2>/dev/null   # путь до lib/... и есть venv
source <путь>/bin/activate
```

```powershell
# Windows: найти venv с torch
Get-ChildItem $HOME -Recurse -Filter torch -Directory -ErrorAction SilentlyContinue | Where-Object FullName -match 'site-packages\\torch$'
<путь>\Scripts\activate
```

```bash
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"   # должно быть True
```

## Обучение

```bash
python train.py --run-dir runs/main
```

Одна итерация: `--games-per-iter` партий self-play → позиции в replay buffer → `ceil(новые позиции * reuse / batch)` шагов обучения → `latest.pt` и `buffer.npz`. Если остановить (Ctrl+C) и запустить снова, обучение продолжится с последнего чекпоинта. Каждые `--save-every` итераций сохраняется `iter_XXXX.pt`.

Основные параметры (дефолты):

| параметр | дефолт | что это |
|---|---|---|
| `--blocks` / `--channels` | 10 / 128 | размер ResNet (~3.4M параметров), у AlphaZero было 20x256 |
| `--simulations` | 200 | симуляций MCTS на ход (у AlphaZero 800) |
| `--parallel-games` | 128 | партий одновременно = размер батча для сети |
| `--games-per-iter` | 256 | партий за итерацию |
| `--buffer-size` / `--min-buffer` | 500k / 20k | окно позиций и минимум до старта обучения |
| `--batch-size` / `--lr` | 1024 / 1e-3 | AdamW, weight decay 1e-4 |
| `--reuse` | 4 | сколько раз в среднем учимся на каждой позиции |

Mixed precision включается само: bf16, а если карта его не поддерживает, fp16 с GradScaler. Отключается флагом `--no-amp`.

Узкое место: MCTS написан на чистом Python и выдаёт около 6k симуляций/с на ядро, так что GPU будет простаивать. Если self-play медленный, сначала уменьши `--simulations` (100 для старта нормально) и `--max-plies`.

## Проверка силы

```bash
python arena.py runs/main/latest.pt random --games 50              # против случайных ходов
python arena.py runs/main/iter_0050.pt runs/main/iter_0020.pt      # прогресс между версиями
```

## Игра

```bash
python uci.py runs/main/latest.pt --nodes 800
```

Добавь эту команду как движок в Cute Chess / Arena / En Croissant. Поддерживается `go nodes`, `go movetime` и `go wtime/btime`. Команду `stop` движок не обрабатывает.

## Устройство

- `azchess/encoding.py`: вход 20 плоскостей 8x8 (фигуры, повторение, рокировки, правило 50 ходов, en passant) и политика 73x8x8 = 4672 хода, как в AlphaZero. Доска всегда повёрнута к стороне, которая ходит.
- `azchess/model.py`: ResNet, conv policy head, value head с tanh.
- `azchess/mcts.py`: PUCT с FPU, шум Дирихле в корне, переиспользование дерева. Деревья всех параллельных партий оцениваются одним батчем.
- `azchess/selfplay.py`: первые 30 полуходов ход выбирается пропорционально визитам, дальше берётся лучший.
- `azchess/replay.py`: кольцевой буфер с разреженными policy-таргетами.

Тесты: `python -m pytest`.
