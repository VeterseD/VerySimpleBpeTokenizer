"""Self-play -> train loop. Resumes automatically from <run-dir>/latest.pt."""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from azchess.encoding import POLICY_SIZE
from azchess.mcts import Evaluator, SearchConfig
from azchess.model import AlphaZeroNet, amp_dtype_for, default_device, load_checkpoint, save_checkpoint
from azchess.replay import ReplayBuffer
from azchess.selfplay import SelfPlayConfig, play_games


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="AlphaZero-style chess training")
    p.add_argument("--run-dir", default="runs/main")
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--blocks", type=int, default=10)
    p.add_argument("--channels", type=int, default=128)
    p.add_argument("--games-per-iter", type=int, default=256)
    p.add_argument("--parallel-games", type=int, default=128, help="games searched together = NN batch size")
    p.add_argument("--simulations", type=int, default=200)
    p.add_argument("--max-plies", type=int, default=400)
    p.add_argument("--temperature-plies", type=int, default=30)
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--buffer-size", type=int, default=500_000)
    p.add_argument("--min-buffer", type=int, default=20_000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--reuse", type=float, default=4.0, help="how many times each position is trained on on average")
    p.add_argument("--train-steps", type=int, default=None, help="fixed steps per iteration (overrides --reuse)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--save-every", type=int, default=10, help="keep iter_XXXX.pt every N iterations")
    p.add_argument("--device", default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def train_steps(model, optimizer, scaler, buffer, steps, batch_size, device, amp_dtype, rng):
    model.train()
    policy_sum = torch.zeros((), device=device)
    value_sum = torch.zeros((), device=device)
    for _ in range(steps):
        planes, pidx, pp, value = buffer.sample(batch_size, rng)
        x = torch.from_numpy(planes).to(device, non_blocking=True)
        idx = torch.from_numpy(pidx.astype(np.int64)).to(device, non_blocking=True)
        p = torch.from_numpy(pp.astype(np.float32)).to(device, non_blocking=True)
        z = torch.from_numpy(value).to(device, non_blocking=True)
        # padded slots have idx -1 and p 0; scatter_add keeps them harmless
        target = torch.zeros(len(x), POLICY_SIZE, device=device).scatter_add_(1, idx.clamp(min=0), p)

        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits, v = model(x)
        policy_loss = -(target * F.log_softmax(logits.float(), dim=1)).sum(1).mean()
        value_loss = F.mse_loss(v.float(), z)
        loss = policy_loss + value_loss

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        policy_sum += policy_loss.detach()
        value_sum += value_loss.detach()
    model.eval()
    return policy_sum.item() / steps, value_sum.item() / steps


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device) if args.device else default_device()
    amp_dtype = None if args.no_amp else amp_dtype_for(device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    latest, buffer_path = run_dir / "latest.pt", run_dir / "buffer.npz"

    if latest.exists():
        model, ckpt = load_checkpoint(latest, device)
        start_iter = ckpt["iteration"]
        print(f"resumed {latest} at iteration {start_iter} ({model.config})", flush=True)
    else:
        model, ckpt, start_iter = AlphaZeroNet(args.blocks, args.channels).to(device), None, 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if ckpt is not None and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)

    buffer = ReplayBuffer(args.buffer_size)
    if buffer_path.exists():
        buffer.load(buffer_path)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} amp={amp_dtype} params={n_params / 1e6:.2f}M buffer={buffer.size}", flush=True)

    sp_cfg = SelfPlayConfig(
        simulations=args.simulations,
        parallel_games=args.parallel_games,
        temperature_plies=args.temperature_plies,
        max_plies=args.max_plies,
        search=SearchConfig(c_puct=args.c_puct),
    )
    evaluator = Evaluator(model, device, amp_dtype)

    for it in range(start_iter + 1, args.iterations + 1):
        model.eval()
        t0 = time.time()
        games, sims = play_games(evaluator, args.games_per_iter, sp_cfg, rng)
        for g in games:
            buffer.add_game(g)
        sp_time = time.time() - t0
        new_positions = sum(g.plies for g in games)
        wins = sum(g.result > 0 for g in games)
        losses = sum(g.result < 0 for g in games)
        print(
            f"[{it}] selfplay: {len(games)} games W/D/L {wins}/{len(games) - wins - losses}/{losses} "
            f"avg {new_positions / len(games):.0f} plies, {sp_time:.0f}s ({sims / sp_time:.0f} sims/s), "
            f"buffer {buffer.size}",
            flush=True,
        )

        if buffer.size >= args.min_buffer:
            steps = args.train_steps or max(1, math.ceil(new_positions * args.reuse / args.batch_size))
            t0 = time.time()
            pl, vl = train_steps(model, optimizer, scaler, buffer, steps, args.batch_size, device, amp_dtype, rng)
            print(f"[{it}] train: {steps} steps policy {pl:.3f} value {vl:.3f} ({time.time() - t0:.0f}s)", flush=True)

        save_checkpoint(latest, model, optimizer, it)
        if it % args.save_every == 0:
            save_checkpoint(run_dir / f"iter_{it:04d}.pt", model, None, it)
        buffer.save(buffer_path)


if __name__ == "__main__":
    main()
