import subprocess
import sys
from pathlib import Path

import chess
import numpy as np

import arena
import train
from azchess.replay import ReplayBuffer

ROOT = Path(__file__).resolve().parent.parent


def test_replay_buffer_wraps_and_roundtrips(tmp_path):
    buf = ReplayBuffer(4)
    n = 6
    planes = np.arange(n, dtype=np.uint8)[:, None, None, None] * np.ones((1, 20, 8, 8), np.uint8)
    idx = np.full((n, 256), -1, np.int16)
    idx[:, 0] = np.arange(n)
    probs = np.zeros((n, 256), np.float16)
    probs[:, 0] = 1
    z = np.array([1, -1, 1, -1, 0, 1], np.float32)
    buf.add_batch(planes[:3], idx[:3], probs[:3], z[:3])
    buf.add_batch(planes[3:], idx[3:], probs[3:], z[3:])
    buf.save(tmp_path / "b.npz")

    small = ReplayBuffer(3)
    small.load(tmp_path / "b.npz")
    assert small.size == 3
    assert [int(small.planes[i, 0, 0, 0]) for i in range(3)] == [3, 4, 5]
    assert list(small.value) == [-1.0, 0.0, 1.0]


def test_train_resume_uci_and_arena(tmp_path, capsys):
    common = [
        "--run-dir", str(tmp_path), "--blocks", "1", "--channels", "16", "--games-per-worker", "8",
        "--simulations", "8", "--full-search-prob", "1", "--max-plies", "30", "--buffer-size", "5000",
        "--min-buffer", "100", "--batch-size", "32", "--device", "cpu", "--no-amp", "--publish-every", "2",
        "--checkpoint-every", "2", "--snapshot-every", "3", "--log-seconds", "1",
    ]
    train.main(common + ["--total-steps", "4"])
    train.main(common + ["--total-steps", "8"])
    out = capsys.readouterr().out
    assert "resumed" in out and "saved" in out and "sims/s" in out
    ckpt = tmp_path / "latest.pt"
    assert list(tmp_path.glob("step_*.pt")) and (tmp_path / "buffer.npz").exists()

    proc = subprocess.run(
        [sys.executable, str(ROOT / "uci.py"), str(ckpt), "--device", "cpu"],
        input="uci\nisready\nposition startpos moves e2e4\ngo nodes 64\ngo movetime 200\n"
        "position fen 6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1\ngo nodes 400\nquit\n",
        capture_output=True, text=True, timeout=120, cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    best = [line.split()[1] for line in proc.stdout.splitlines() if line.startswith("bestmove")]
    board = chess.Board()
    board.push_uci("e2e4")
    assert len(best) == 3 and all(chess.Move.from_uci(m) in board.legal_moves for m in best[:2])
    assert best[2] == "a1a8"

    arena.main([str(ckpt), "random", "--games", "4", "--simulations", "8", "--max-plies", "30", "--device", "cpu"])
    assert "A vs B: +" in capsys.readouterr().out
