#!/usr/bin/env python3
"""Profile two Qwen-shaped MoE layers with separate expert FSDP and EP=1."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import os
from pathlib import Path
from time import time

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
import torch.nn.functional as F

from torchtitan.models.qwen3.model.model import Qwen3Model

from profile_torchtitan_qwen3 import (
    build_model_args,
    build_parallel_dims,
    configure_dependency_ordered_prefetch,
    describe_prefetch_graph,
    git_revision,
    nvtx_range,
    register_layer_ranges,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--emit-torch-nvtx", action="store_true")
    parser.add_argument("--module-labels", action="store_true")
    parser.add_argument("--perfetto-trace-dir", type=Path)
    parser.add_argument(
        "--torchtitan-root",
        type=Path,
        default=Path(os.environ.get("TORCHTITAN_ROOT", ".")),
    )
    return parser.parse_args()


def apply_separate_expert_fsdp(
    model: Qwen3Model, dp_mesh: torch.distributed.device_mesh.DeviceMesh
) -> None:
    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
    fsdp_config = {"mesh": dp_mesh, "mp_policy": policy}

    fully_shard(model.tok_embeddings, **fsdp_config, reshard_after_forward=True)
    for layer in model.layers.values():
        fully_shard(layer.moe.experts, **fsdp_config, reshard_after_forward=True)
        fully_shard(layer, **fsdp_config, reshard_after_forward=True)
    fully_shard(
        [model.norm, model.output],
        **fsdp_config,
        reshard_after_forward=False,
    )
    fully_shard(model, **fsdp_config)
    configure_dependency_ordered_prefetch(model, backward_experts_first=True)


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)

    if world_size != 8:
        raise ValueError("This benchmark requires exactly eight GPUs")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    model_args = build_model_args(args)
    parallel_dims = build_parallel_dims("fsdp", world_size, None)
    dp_mesh = parallel_dims.world_mesh["dp_shard_cp"]

    with torch.device("meta"):
        model = Qwen3Model(model_args)
    apply_separate_expert_fsdp(model, dp_mesh)
    model.to_empty(device=device)
    with torch.no_grad():
        model.init_weights(buffer_device=device)
    model.train()
    if args.module_labels:
        register_layer_ranges(model)
    prefetch_graph = describe_prefetch_graph(model)

    tokens = torch.arange(
        args.batch * args.sequence, device=device, dtype=torch.long
    ).reshape(args.batch, args.sequence) % args.vocab_size
    targets = torch.roll(tokens, shifts=-1, dims=1)

    def train_step() -> float:
        model.zero_grad(set_to_none=True)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with nvtx_range("Train/forward"):
            logits = model(tokens)
        with nvtx_range("Train/loss"):
            loss = F.cross_entropy(
                logits.float().reshape(-1, args.vocab_size), targets.reshape(-1)
            )
        with nvtx_range("Train/backward"):
            loss.backward()
        end.record()
        end.synchronize()
        return start.elapsed_time(end)

    for _ in range(args.warmup):
        train_step()
    dist.barrier()

    if args.capture:
        torch.cuda.cudart().cudaProfilerStart()
    if args.emit_torch_nvtx and args.perfetto_trace_dir:
        raise ValueError("Use separate runs for NVTX and Perfetto output")
    nvtx_context = (
        torch.autograd.profiler.emit_nvtx(record_shapes=False)
        if args.emit_torch_nvtx
        else nullcontext()
    )
    should_profile = args.perfetto_trace_dir is not None and rank == 0
    profiler_context = (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )
        if should_profile
        else nullcontext()
    )
    with nvtx_context, profiler_context as profiler:
        step_times = [train_step() for _ in range(args.steps)]
    torch.cuda.synchronize(device)
    if args.capture:
        torch.cuda.cudart().cudaProfilerStop()
    dist.barrier()

    if should_profile:
        args.perfetto_trace_dir.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(args.perfetto_trace_dir / "rank-0.json"))

    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        script_path = Path(__file__).resolve()
        support_path = script_path.with_name("profile_torchtitan_qwen3.py")
        summary = {
            "schema_version": 1,
            "timestamp_unix": time(),
            "scenario": "fsdp_moe_ep1",
            "implementation": "TorchTitan Qwen3 MoE with separate expert FSDP2",
            "torchtitan_revision": git_revision(args.torchtitan_root),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "nccl_version": ".".join(map(str, torch.cuda.nccl.version())),
            "gpu": torch.cuda.get_device_name(device),
            "world_size": world_size,
            "model": {
                "source_flavor": "Qwen3-235B-A22B",
                "layers": 2,
                "vocab_size": args.vocab_size,
                "dim": model_args.dim,
                "heads": model_args.n_heads,
                "kv_heads": model_args.n_kv_heads,
                "head_dim": model_args.head_dim,
                "experts": model_args.moe_args.num_experts,
                "active_experts": model_args.moe_args.top_k,
                "expert_intermediate_dim": model_args.moe_inter_dim,
                "grouped_mm": model_args.moe_args.use_grouped_mm,
                "balanced_synthetic_routing": model_args.moe_args._debug_force_load_balance,
            },
            "workload": {
                "batch_per_rank": args.batch,
                "sequence": args.sequence,
                "synthetic_tokens": True,
                "loss": "cross_entropy",
            },
            "parallel_dims": {
                key: value
                for key, value in asdict(parallel_dims).items()
                if key != "_world_mesh"
            },
            "expert_parallel_degree": 1,
            "expert_fsdp_degree": world_size,
            "prefetch_policy": "dependency_ordered_experts_first_backward",
            "prefetch_graph": prefetch_graph,
            "module_labels": args.module_labels,
            "mean_step_ms": sum(step_times) / len(step_times),
            "step_ms": step_times,
        }
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print(json.dumps(summary, indent=2))

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
