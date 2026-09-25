"""Self-play worker process: native MCTS (azchess_rs) + GPU inference, pipelined over two game groups."""
from __future__ import annotations

import os
import queue as queue_mod
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

import azchess_rs

from .inference import Evaluator, inference_dtype


@dataclass
class SelfPlayConfig:
    games: int = 512  # concurrent games per worker, split into two pipelined groups
    simulations: int = 200
    fast_simulations: int = 50
    full_search_prob: float = 0.25
    temperature_plies: int = 30
    max_plies: int = 400
    c_puct: float = 1.5
    fpu_reduction: float = 0.25
    dirichlet_alpha: float = 0.3
    dirichlet_frac: float = 0.25

    def make_group(self, num_games: int, seed: int) -> "azchess_rs.SelfPlay":
        kwargs = asdict(self)
        del kwargs["games"]
        return azchess_rs.SelfPlay(num_games=num_games, seed=seed, **kwargs)


class SelfPlayRunner:
    """Two groups ping-pong: the CPU searches one group while the GPU evaluates the other."""

    def __init__(self, evaluator: Evaluator, cfg: SelfPlayConfig, seed: int):
        self.evaluator = evaluator
        sizes = [cfg.games - cfg.games // 2, cfg.games // 2] if cfg.games > 1 else [1]
        self.groups = [cfg.make_group(n, seed * 1_000 + i) for i, n in enumerate(sizes)]
        self.handles = [evaluator.launch(g.collect()) for g in self.groups]

    def step(self) -> None:
        for i, group in enumerate(self.groups):
            logits, values = self.evaluator.wait(self.handles[i])
            group.apply(logits, values)
            self.handles[i] = self.evaluator.launch(group.collect())

    @property
    def total_sims(self) -> int:
        return sum(g.total_sims for g in self.groups)

    def take_finished(self):
        """(planes, idx, probs f16, z, results, plies, movetext list, termination list) or None."""
        parts = [g.take_finished() for g in self.groups if g.num_finished]
        if not parts:
            return None
        columns = list(zip(*parts))
        planes, idx, probs, z, results, plies = (np.concatenate(x) for x in columns[:6])
        movetext = [m for part in columns[6] for m in part]
        terminations = [t for part in columns[7] for t in part]
        return planes, idx, probs.astype(np.float16), z, results, plies, movetext, terminations


def load_weights(path: str) -> dict:
    for attempt in range(20):
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, EOFError):
            if attempt == 19:
                raise
            time.sleep(0.2)


def selfplay_worker(
    worker_id: int,
    cfg: SelfPlayConfig,
    model_config: dict,
    device: str,
    amp: bool,
    weights_path: str,
    version,
    out_queue,
    stop,
    seed: int,
    send_every: float = 2.0,
) -> None:
    out_queue.cancel_join_thread()  # exit promptly on shutdown even if the trainer stopped reading
    try:
        dev = torch.device(device)
        if dev.type == "cuda":
            torch.set_num_threads(1)
        else:
            torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
        evaluator = Evaluator(model_config, dev, inference_dtype(dev, amp))
        loaded = version.value
        evaluator.load_state_dict(load_weights(weights_path))
        runner = SelfPlayRunner(evaluator, cfg, seed)
        last_send, last_sims = time.monotonic(), 0
        while not stop.is_set():
            runner.step()
            if version.value != loaded:
                loaded = version.value
                evaluator.load_state_dict(load_weights(weights_path))
            now = time.monotonic()
            if now - last_send >= send_every:
                sims = runner.total_sims
                finished = runner.take_finished()
                try:
                    out_queue.put(("games", worker_id, finished, sims - last_sims, loaded), timeout=60)
                except queue_mod.Full:
                    continue
                last_send, last_sims = now, sims
    except KeyboardInterrupt:
        pass
