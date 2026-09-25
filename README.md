# azchess

Шахматный движок в стиле AlphaZero. Сеть ResNet (policy + value) на PyTorch/CUDA, а MCTS, генерация ходов и кодирование доски написаны на Rust ([shakmaty](https://crates.io/crates/shakmaty) + PyO3 + rayon). Python только гоняет сеть и обучение.

## Установка

```powershell
F:\venvs\torch\Scripts\Activate.ps1
cd <папка с репой>

# что уже стоит
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
pip list | Select-String -Pattern "^(torch|numpy|chess|maturin|pytest) "
rustc --version
```

Ставь только то, чего нет в выводе. `pip install` пропускает пакеты, которые уже стоят, torch он не трогает.

```powershell
pip install chess maturin pytest          # только отсутствующие
pip install --no-deps .\azchess_rs         # сборка Rust-модуля, зависимости не трогает
```

Для сборки нужен Rust (https://rustup.rs) и на Windows MSVC Build Tools (C++). Если меняешь Rust-код: `maturin develop --release -m azchess_rs/Cargo.toml`.

## Обучение

```bash
python train.py --run-dir runs/main
```

Обучение асинхронное, как в Lc0/KataGo:

- **Self-play процесс** держит 512 партий одновременно. Rust на всех ядрах спускается по деревьям и отдаёт батч листьев, сеть оценивает его на GPU в bf16/fp16. Партии разбиты на две группы: пока GPU считает одну, CPU ищет по другой.
- **Тренер** (основной процесс) параллельно учится на replay buffer, каждые `--publish-every` шагов отдаёт свежие веса в self-play и держит соотношение: в среднем `--reuse` обучающих примеров на одну новую позицию.
- **Playout cap randomization** (KataGo): 25% ходов ищутся на `--simulations 200` и идут в обучение, остальные делаются быстро на `--fast-simulations 50` и не записываются. Партий генерируется в несколько раз больше за то же время. Отключается флагом `--full-search-prob 1`.

Ctrl+C сохраняет `latest.pt` и `buffer.npz`, следующий запуск продолжит с того же места. Каждые `--snapshot-every` шагов сохраняется `step_XXXXXXX.pt` для сравнения версий.

Раз в 30 секунд печатается строка статистики:
```
[step 1200] 41,000 sims/s 1,900 pos/s 9.8 games/s | W/D/L 31/40/29% avg 190 plies | 7.4 steps/s policy 2.810 value 0.612 | buffer 310,000 games 20,400
```

Главные параметры:

| параметр | дефолт | что это |
|---|---|---|
| `--blocks` / `--channels` | 10 / 128 | размер ResNet (~3.4M параметров), у AlphaZero было 20x256 |
| `--games-per-worker` | 512 | партий одновременно, половина из них = размер батча сети |
| `--workers` | 1 | процессов self-play. Rust и так занимает все ядра, 2 могут помочь, если GPU недогружен |
| `--simulations` / `--fast-simulations` / `--full-search-prob` | 200 / 50 / 0.25 | см. выше |
| `--buffer-size` / `--min-buffer` | 1M / 50k | окно позиций и минимум до старта обучения |
| `--batch-size` / `--lr` / `--reuse` | 1024 / 1e-3 / 4 | AdamW, weight decay 1e-4 |

**Если self-play медленный** (мало `sims/s`, GPU загружен не полностью), подними `--games-per-worker` или поставь `--workers 2`. **Если GPU загружен на 100%**, это нормально: упираемся в сеть. Ускоряют либо сеть поменьше, либо меньше `--simulations`.

## Проверка силы

```bash
python arena.py runs/main/latest.pt random --games 50
python arena.py runs/main/step_0050000.pt runs/main/step_0020000.pt --games 200
```

## Игра

```bash
python uci.py runs/main/latest.pt --nodes 800
```

Добавь эту команду как движок в Cute Chess / Arena / En Croissant. Поддерживаются `go nodes`, `go movetime` и `go wtime/btime`. Поиск батчит по 16 листьев за вызов сети (virtual loss, `--batch`).

## Устройство

- `azchess_rs/src/lib.rs`: кодирование (20 плоскостей 8x8), политика 73x8x8 = 4672 хода как в AlphaZero, PUCT с FPU, шум Дирихле, virtual loss, переиспользование дерева, `SelfPlay` (партии для обучения) и `Searcher` (игра и arena).
- `azchess/model.py`: ResNet с conv policy head и value head на tanh.
- `azchess/inference.py`: асинхронные вызовы сети (pinned memory, CUDA events).
- `azchess/selfplay.py`: процесс self-play. `train.py`: тренер.
- `azchess/encoding.py`: эталонная реализация кодирования на python-chess. Тесты сверяют с ней Rust-код.

Тесты: `python -m pytest`.
