#!/usr/bin/env python3
"""Measure NCCL AllGather and ReduceScatter latency over relevant payloads."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes-mb", default="16,64,142,604")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("results/collectives.json"))
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    results = []

    for size_mb in [float(value) for value in args.sizes_mb.split(",")]:
        shard_elements = int(size_mb * 1024 * 1024 / 2)
        shard = torch.empty(shard_elements, dtype=torch.bfloat16, device=device)
        gathered = torch.empty(shard_elements * world_size, dtype=torch.bfloat16, device=device)
        reduce_input = torch.empty_like(gathered)
        reduced = torch.empty_like(shard)

        def measure(callback) -> float:
            for _ in range(args.warmup):
                callback()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.steps):
                callback()
            end.record()
            end.synchronize()
            return start.elapsed_time(end) / args.steps

        results.append({
            "shard_mb": size_mb,
            "aggregate_mb": size_mb * world_size,
            "all_gather_ms": measure(lambda: dist.all_gather_into_tensor(gathered, shard)),
            "reduce_scatter_ms": measure(lambda: dist.reduce_scatter_tensor(reduced, reduce_input)),
        })

    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {"world_size": world_size, "dtype": "bf16", "results": results}
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
