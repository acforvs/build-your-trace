#!/usr/bin/env python3
"""Profile a single Qwen-shaped dense training layer on one H100."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
from time import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


@contextmanager
def nvtx_range(message: str):
    torch.cuda.nvtx.range_push(message)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(x.dtype)


class QwenAttentionBlock(nn.Module):
    def __init__(self, dim: int, query_heads: int, kv_heads: int, head_dim: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(dim, query_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(query_heads * head_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = x.shape
        normalized = self.norm(x)
        query = self.q_proj(normalized).view(
            batch, sequence, self.query_heads, self.head_dim
        ).transpose(1, 2)
        key = self.k_proj(normalized).view(
            batch, sequence, self.kv_heads, self.head_dim
        ).transpose(1, 2)
        value = self.v_proj(normalized).view(
            batch, sequence, self.kv_heads, self.head_dim
        ).transpose(1, 2)
        heads_per_kv = self.query_heads // self.kv_heads
        key = key.repeat_interleave(heads_per_kv, dim=1)
        value = value.repeat_interleave(heads_per_kv, dim=1)
        attended = F.scaled_dot_product_attention(
            query, key, value, is_causal=True
        )
        attended = attended.transpose(1, 2).reshape(
            batch, sequence, self.query_heads * self.head_dim
        )
        return x + self.o_proj(attended)


class QwenMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(x)
        return x + self.down_proj(F.silu(self.gate_proj(normalized)) * self.up_proj(normalized))


def register_backward_range(module: nn.Module, name: str) -> None:
    records = []

    def pre_hook(_module, _grad_output):
        torch.cuda.nvtx.range_push(name)
        records.append(name)

    def hook(_module, _grad_input, _grad_output):
        records.pop()
        torch.cuda.nvtx.range_pop()

    module.register_full_backward_pre_hook(pre_hook)
    module.register_full_backward_hook(hook)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture", action="store_true")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    if dist.get_world_size() != 1:
        raise ValueError("This benchmark requires exactly one GPU")
    device = torch.device("cuda", local_rank)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    attention = QwenAttentionBlock(4096, 64, 4, 128).to(
        device=device, dtype=torch.bfloat16
    )
    mlp = QwenMLPBlock(4096, 12288).to(device=device, dtype=torch.bfloat16)
    register_backward_range(mlp, "Dense/MLP backward")
    register_backward_range(attention, "Dense/attention backward")
    parameters = list(attention.parameters()) + list(mlp.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, foreach=True)

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    hidden_states = torch.randn(
        args.batch,
        args.sequence,
        4096,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    hidden_states.requires_grad_(True)

    def train_step() -> float:
        optimizer.zero_grad(set_to_none=True)
        hidden_states.grad = None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with nvtx_range("Dense/attention forward"):
            output = attention(hidden_states)
        with nvtx_range("Dense/MLP forward"):
            output = mlp(output)
        with nvtx_range("Dense/loss"):
            loss = output.float().square().mean()
        with nvtx_range("Dense/backward"):
            loss.backward()
        with nvtx_range("Dense/optimizer step"):
            optimizer.step()
        end.record()
        end.synchronize()
        return start.elapsed_time(end)

    for _ in range(args.warmup):
        train_step()
    dist.barrier()
    if args.capture:
        torch.cuda.cudart().cudaProfilerStart()
    step_times = [train_step() for _ in range(args.steps)]
    torch.cuda.synchronize(device)
    if args.capture:
        torch.cuda.cudart().cudaProfilerStop()
    dist.barrier()

    args.output.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve()
    summary = {
        "schema_version": 1,
        "timestamp_unix": time(),
        "implementation": "Explicit Qwen-shaped GQA + dense SwiGLU",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "nccl_version": ".".join(map(str, torch.cuda.nccl.version())),
        "gpu": torch.cuda.get_device_name(device),
        "world_size": 1,
        "model": {
            "hidden_size": 4096,
            "query_heads": 64,
            "kv_heads": 4,
            "head_dim": 128,
            "dense_intermediate_size": 12288,
            "parameter_dtype": "BF16",
        },
        "workload": {
            "batch": args.batch,
            "sequence": args.sequence,
            "synthetic_hidden_states": True,
            "loss": "mean squared activation",
            "optimizer": "AdamW foreach",
        },
        "mean_step_ms": sum(step_times) / len(step_times),
        "step_ms": step_times,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
