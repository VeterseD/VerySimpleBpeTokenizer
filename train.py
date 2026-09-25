"""Asynchronous AlphaZero training: self-play worker processes stream games, this process trains.

Resumes automatically from <run-dir>/latest.pt and <run-dir>/buffer.npz. Stop with Ctrl+C.
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import queue as queue_mod
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from azchess.encoding import POLICY_SIZE
from azchess.model import AlphaZeroNet, amp_dtype_for, default_device, save_checkpoint
from azchess.replay import ReplayBuffer
from azchess.selfplay import SelfPlayConfig, selfplay_worker


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="AlphaZero-style chess training")
    p.add_argument("--run-dir", default="runs/main")
    p.add_argument("--blocks", type=int, default=10)
    p.add_argument("--channels", type=int, default=128)
    # self-play
    p.add_argument("--workers", type=int, default=1, help="self-play processes (each uses all CPU cores via rayon)")
    p.add_argument("--games-per-worker", type=int, default=512, help="concurrent games per worker (~NN batch x2)")
    p.add_argument("--simulations", type=int, default=200, help="MCTS simulations for a full (recorded) search")
    p.add_argument("--fast-simulations", type=int, default=50, help="simulations for fast (unrecorded) moves")
    p.add_argument("--full-search-prob", type=float, default=0.25, help="share of moves searched fully (1 = off)")
    p.add_argument("--temperature-plies", type=int, default=30)
    p.add_argument("--max-plies", type=int, default=400)
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--selfplay-device", default=None, help="default: same as --device")
    # training
    p.add_argument("--buffer-size", type=int, default=1_000_000)
    p.add_argument("--min-buffer", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--reuse", type=float, default=4.0, help="average number of times each position is trained on")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--total-steps", type=int, default=None, help="stop after this many training steps")
    p.add_argument("--publish-every", type=int, default=200, help="send new weights to self-play every N steps")
    p.add_argument("--checkpoint-every", type=int, default=1000)
    p.add_argument("--snapshot-every", type=int, default=10_000, help="keep step_XXXXXXX.pt every N steps")
    p.add_argument("--buffer-save-minutes", type=float, default=30.0)
    p.add_argument("--log-seconds", type=float, default=30.0)
    p.add_argument("--device", default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


class Trainer:
    def __init__(self, model, lr, weight_decay, device, amp_dtype):
        self.model = model
        self.device = device
        self.amp_dtype = amp_dtype
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)
        self.policy_sum = torch.zeros((), device=device)
        self.value_sum = torch.zeros((), device=device)
        self.loss_steps = 0

    def _to_device(self, array: np.ndarray, dtype=None) -> torch.Tensor:
        t = torch.from_numpy(array)
        if self.device.type == "cuda":
            t = t.pin_memory()
        return t.to(self.device, dtype=dtype, non_blocking=True)

    def step(self, batch) -> None:
        planes, pidx, pp, value = batch
        x = self._to_device(planes)
        idx = self._to_device(pidx, torch.int64)
        p = self._to_device(pp, torch.float32)
        z = self._to_device(value)
        # padded slots have idx -1 and p 0; scatter_add keeps them harmless
        target = torch.zeros(len(x), POLICY_SIZE, device=self.device).scatter_add_(1, idx.clamp(min=0), p)

        with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
            logits, v = self.model(x)
        policy_loss = -(target * F.log_softmax(logits.float(), dim=1)).sum(1).mean()
        value_loss = F.mse_loss(v.float(), z)

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(policy_loss + value_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.policy_sum += policy_loss.detach()
        self.value_sum += value_loss.detach()
        self.loss_steps += 1

    def pop_losses(self) -> tuple[float, float] | None:
        if self.loss_steps == 0:
            return None
        out = (self.policy_sum.item() / self.loss_steps, self.value_sum.item() / self.loss_steps)
        self.policy_sum.zero_()
        self.value_sum.zero_()
        self.loss_steps = 0
        return out


class Stats:
    def __init__(self):
        self.reset()

    def reset(self):
        self.t0 = time.monotonic()
        self.positions = self.games = self.sims = self.plies = self.steps = 0
        self.wdl = [0, 0, 0]

    def add_games(self, results: np.ndarray, plies: np.ndarray, positions: int):
        self.games += len(results)
        self.plies += int(plies.sum())
        self.positions += positions
        self.wdl[0] += int((results > 0).sum())
        self.wdl[1] += int((results == 0).sum())
        self.wdl[2] += int((results < 0).sum())


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device) if args.device else default_device()
    selfplay_device = args.selfplay_device or str(device)
    amp_dtype = None if args.no_amp else amp_dtype_for(device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    latest, buffer_path, weights_path = run_dir / "latest.pt", run_dir / "buffer.npz", run_dir / "weights.pt"

    ckpt = torch.load(latest, map_location="cpu", weights_only=True) if latest.exists() else None
    model_config = ckpt["model_config"] if ckpt else {"blocks": args.blocks, "channels": args.channels}
    model = AlphaZeroNet(**model_config).to(device)
    trainer = Trainer(model, args.lr, args.weight_decay, device, amp_dtype)
    step = total_positions = total_games = 0
    if ckpt:
        model.load_state_dict(ckpt["model"])
        if ckpt.get("optimizer"):
            trainer.optimizer.load_state_dict(ckpt["optimizer"])
        step, total_positions, total_games = ckpt["step"], ckpt["total_positions"], ckpt["total_games"]
        print(f"resumed {latest} at step {step} ({model_config})", flush=True)
    model.train()

    buffer = ReplayBuffer(args.buffer_size)
    if buffer_path.exists():
        buffer.load(buffer_path)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} selfplay={selfplay_device} amp={amp_dtype} params={n_params / 1e6:.2f}M "
          f"buffer={buffer.size}", flush=True)

    def checkpoint(path=latest, with_optimizer=True):
        save_checkpoint(path, {
            "model": model.state_dict(),
            "model_config": model_config,
            "optimizer": trainer.optimizer.state_dict() if with_optimizer else None,
            "step": step,
            "total_positions": total_positions,
            "total_games": total_games,
        })

    def publish():
        save_checkpoint(weights_path, model.state_dict())
        version.value += 1

    ctx = mp.get_context("spawn")
    version = ctx.Value("q", 0)
    stop = ctx.Event()
    games_queue = ctx.Queue(maxsize=64)
    publish()
    sp_cfg = SelfPlayConfig(
        games=args.games_per_worker,
        simulations=args.simulations,
        fast_simulations=args.fast_simulations,
        full_search_prob=args.full_search_prob,
        temperature_plies=args.temperature_plies,
        max_plies=args.max_plies,
        c_puct=args.c_puct,
    )
    workers = [
        ctx.Process(
            target=selfplay_worker,
            args=(i, sp_cfg, model_config, selfplay_device, not args.no_amp, str(weights_path), version,
                  games_queue, stop, args.seed * 100 + i + step),
            daemon=True,
        )
        for i in range(args.workers)
    ]
    for w in workers:
        w.start()

    stats = Stats()
    last_publish = last_ckpt = last_snapshot = step
    last_buffer_save = time.monotonic()
    losses = None

    def handle(msg):
        nonlocal total_positions, total_games
        _, _, finished, sims, _ = msg
        stats.sims += sims
        if finished is None:
            return
        planes, idx, probs, z, results, plies = finished
        buffer.add_batch(planes, idx, probs, z)
        total_positions += len(z)
        total_games += len(results)
        stats.add_games(results, plies, len(z))

    try:
        while args.total_steps is None or step < args.total_steps:
            while True:
                try:
                    handle(games_queue.get_nowait())
                except queue_mod.Empty:
                    break
            dead = [w for w in workers if not w.is_alive()]
            if dead:
                raise RuntimeError(f"self-play worker exited with code {dead[0].exitcode}")

            target = math.floor(max(0, total_positions - args.min_buffer) * args.reuse / args.batch_size)
            if buffer.size >= args.min_buffer and step < target:
                for _ in range(min(target - step, 50)):
                    trainer.step(buffer.sample(args.batch_size, rng))
                    step += 1
                    stats.steps += 1
                    if args.total_steps is not None and step >= args.total_steps:
                        break
            else:
                try:
                    handle(games_queue.get(timeout=0.5))
                except queue_mod.Empty:
                    pass

            if step - last_publish >= args.publish_every:
                publish()
                last_publish = step
            if step - last_ckpt >= args.checkpoint_every:
                checkpoint()
                last_ckpt = step
            if args.snapshot_every and step // args.snapshot_every > last_snapshot // args.snapshot_every:
                checkpoint(run_dir / f"step_{step:07d}.pt", with_optimizer=False)
                last_snapshot = step
            if time.monotonic() - last_buffer_save >= args.buffer_save_minutes * 60:
                buffer.save(buffer_path)
                last_buffer_save = time.monotonic()

            dt = time.monotonic() - stats.t0
            if dt >= args.log_seconds:
                losses = trainer.pop_losses() or losses
                g = max(stats.games, 1)
                w, d, l = (100 * x / g for x in stats.wdl)
                loss_str = f"policy {losses[0]:.3f} value {losses[1]:.3f}" if losses else "waiting for min-buffer"
                print(
                    f"[step {step}] {stats.sims / dt:,.0f} sims/s {stats.positions / dt:,.0f} pos/s "
                    f"{stats.games / dt:.1f} games/s | W/D/L {w:.0f}/{d:.0f}/{l:.0f}% avg {stats.plies / g:.0f} plies | "
                    f"{stats.steps / dt:.1f} steps/s {loss_str} | buffer {buffer.size:,} games {total_games:,}",
                    flush=True,
                )
                stats.reset()
    except KeyboardInterrupt:
        print("stopping...", flush=True)
    finally:
        stop.set()
        checkpoint()
        buffer.save(buffer_path)
        for w in workers:
            w.join(timeout=10)
            if w.is_alive():
                w.terminate()
        print(f"saved {latest} at step {step}", flush=True)


if __name__ == "__main__":
    main()
