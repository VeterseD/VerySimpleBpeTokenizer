"""Network inference for search: numpy planes in, numpy logits/values out, with async GPU launches."""
from __future__ import annotations

import numpy as np
import torch

from .encoding import POLICY_SIZE
from .model import AlphaZeroNet

PAD_TO = 32  # fewer distinct batch shapes for cuDNN


def inference_dtype(device: torch.device, amp: bool = True) -> torch.dtype:
    if device.type != "cuda" or not amp:
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


class Evaluator:
    """Owns a separate inference copy of the network (possibly bf16/fp16)."""

    def __init__(self, model_config: dict, device: torch.device, dtype: torch.dtype = torch.float32):
        self.device = device
        self.dtype = dtype
        self.model = AlphaZeroNet(**model_config).to(device=device, dtype=dtype).eval()

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state)

    @torch.inference_mode()
    def launch(self, planes: np.ndarray):
        """Queues the forward pass and returns a handle; on CUDA this does not wait for the GPU."""
        n = len(planes)
        if n == 0:
            return None
        padded = -(-n // PAD_TO) * PAD_TO
        if padded != n:
            planes = np.concatenate([planes, np.zeros((padded - n, *planes.shape[1:]), planes.dtype)])
        x = torch.from_numpy(planes)
        if self.device.type == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
        logits, values = self.model(x)
        logits, values = logits[:n].float(), values[:n].float()
        if self.device.type != "cuda":
            return logits.numpy(), values.numpy(), None
        out_l = torch.empty(logits.shape, dtype=torch.float32, pin_memory=True)
        out_v = torch.empty(values.shape, dtype=torch.float32, pin_memory=True)
        out_l.copy_(logits, non_blocking=True)
        out_v.copy_(values, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        return out_l, out_v, event

    @staticmethod
    def wait(handle) -> tuple[np.ndarray, np.ndarray]:
        if handle is None:
            return np.zeros((0, POLICY_SIZE), np.float32), np.zeros(0, np.float32)
        logits, values, event = handle
        if event is None:
            return logits, values
        event.synchronize()
        return logits.numpy(), values.numpy()

    def __call__(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.wait(self.launch(planes))


def load_evaluator(path, device: torch.device, amp: bool = True) -> Evaluator:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    evaluator = Evaluator(ckpt["model_config"], device, inference_dtype(device, amp))
    evaluator.load_state_dict(ckpt["model"])
    return evaluator


def run_search(searcher, evaluator: Evaluator, simulations: int, slots=None, deadline=None, clock=None) -> None:
    """Drives an azchess_rs.Searcher until the requested simulations are done (or the deadline passes)."""
    searcher.start(simulations, slots)
    while searcher.active():
        planes = searcher.collect()
        logits, values = evaluator(planes)
        searcher.apply(logits, values)
        if deadline is not None and clock() >= deadline:
            searcher.stop()
