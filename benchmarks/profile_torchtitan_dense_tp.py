#!/usr/bin/env python3
"""Profile two dense Qwen-shaped layers with pure TP or TorchTitan TP+SP."""

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
import torch.nn.functional as F
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)
from torch.distributed.tensor.parallel import loss_parallel
from torchtitan.distributed import ParallelDims
from torchtitan.models.llama4.infra.parallelize import apply_compile
from torchtitan.models.qwen3.infra.parallelize import parallelize_qwen3
from torchtitan.models.qwen3.model.args import Qwen3ModelArgs
from torchtitan.models.qwen3.model.model import Qwen3Model

from profile_torchtitan_dense_fsdp import build_model_args, register_dense_ranges
from profile_torchtitan_qwen3 import build_job_config, git_revision, nvtx_range


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--mode", choices=("tp", "tp-sp"), default="tp-sp")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--async-tp", action="store_true")
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


def build_tp_dims(world_size: int) -> ParallelDims:
    return ParallelDims(
        dp_replicate=1,
        dp_shard=1,
        cp=1,
        tp=world_size,
        pp=1,
        ep=1,
        etp=1,
        world_size=world_size,
    )


def apply_pure_tp(model: Qwen3Model, parallel_dims: ParallelDims) -> Qwen3Model:
    tp_mesh = parallel_dims.world_mesh["tp"]
    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Replicate(),
            ),
            "output": ColwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(-1),
                use_local_output=False,
            ),
        },
    )
    for layer in model.layers.values():
        parallelize_module(
            layer,
            tp_mesh,
            {
                "attention": PrepareModuleInput(
                    input_layouts=(Replicate(), Replicate(), None),
                    desired_input_layouts=(Replicate(), Replicate(), None),
                ),
                "attention.wq": ColwiseParallel(use_local_output=False),
                "attention.wk": ColwiseParallel(use_local_output=False),
                "attention.wv": ColwiseParallel(use_local_output=False),
                "attention.q_norm": SequenceParallel(sequence_dim=2),
                "attention.k_norm": SequenceParallel(sequence_dim=2),
                "attention.wo": RowwiseParallel(output_layouts=Replicate()),
                "feed_forward": PrepareModuleInput(
                    input_layouts=(Replicate(),),
                    desired_input_layouts=(Replicate(),),
                ),
                "feed_forward.w1": ColwiseParallel(use_local_output=False),
                "feed_forward.w2": RowwiseParallel(output_layouts=Replicate()),
                "feed_forward.w3": ColwiseParallel(use_local_output=False),
            },
        )
    return model


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)

    if args.sequence % world_size != 0:
        raise ValueError("Sequence length must be divisible by TP degree")
    if args.async_tp and (args.mode != "tp-sp" or not args.compile_model):
        raise ValueError("Async TP requires --mode tp-sp --compile-model")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    model_args: Qwen3ModelArgs = build_model_args(args)
    parallel_dims = build_tp_dims(world_size)
    job_config = build_job_config(args, parallel_dims)
    job_config.compile.enable = args.compile_model
    job_config.compile.components = ["model"]
    job_config.parallelism.enable_async_tensor_parallel = args.async_tp

    with torch.device("meta"):
        model = Qwen3Model(model_args)
    if args.mode == "tp-sp":
        model = parallelize_qwen3(model, parallel_dims, job_config)
    else:
        model = apply_pure_tp(model, parallel_dims)
        if args.compile_model:
            apply_compile(model, job_config.compile)
    model.to_empty(device=device)
    model.to(dtype=torch.bfloat16)
    with torch.no_grad():
        model.init_weights(buffer_device=device)
    model.train()
    if args.module_labels:
        register_dense_ranges(model)

    tokens = torch.arange(
        args.batch * args.sequence, device=device, dtype=torch.long
    ).reshape(args.batch, args.sequence) % args.vocab_size
    targets = torch.roll(tokens, shifts=-1, dims=1)

    def train_step() -> float:
        model.zero_grad(set_to_none=True)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with loss_parallel():
            with nvtx_range("Train/forward"):
                logits = model(tokens)
            with nvtx_range("Train/loss"):
                loss = F.cross_entropy(
                    logits.float().reshape(-1, args.vocab_size),
                    targets.reshape(-1),
                )
            del logits
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
        dense_support = script_path.with_name("profile_torchtitan_dense_fsdp.py")
        common_support = script_path.with_name("profile_torchtitan_qwen3.py")
        summary = {
            "schema_version": 1,
            "timestamp_unix": time(),
            "scenario": f"dense_{args.mode.replace('-', '_')}",
            "implementation": (
                "TorchTitan dense Qwen3 TP with sequence parallelism"
                if args.mode == "tp-sp"
                else "TorchTitan-derived dense Qwen3 pure tensor parallelism"
            ),
            "torchtitan_revision": git_revision(args.torchtitan_root),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "nccl_version": ".".join(map(str, torch.cuda.nccl.version())),
            "gpu": torch.cuda.get_device_name(device),
            "world_size": world_size,
            "model": {
                "shape_reference": "Qwen3-235B-A22B attention width",
                "layers": 2,
                "vocab_size": args.vocab_size,
                "dim": model_args.dim,
                "heads": model_args.n_heads,
                "kv_heads": model_args.n_kv_heads,
                "head_dim": model_args.head_dim,
                "dense_intermediate_dim": model_args.hidden_dim,
                "parameter_dtype": "bfloat16",
            },
            "workload": {
                "batch": args.batch,
                "global_sequence": args.sequence,
                "activation_sequence_per_rank": (
                    args.sequence // world_size
                    if args.mode == "tp-sp"
                    else args.sequence
                ),
                "residual_layout": (
                    "sequence-sharded" if args.mode == "tp-sp" else "replicated"
                ),
                "synthetic_tokens": True,
                "loss": "loss-parallel cross_entropy",
            },
            "parallel_dims": {
                key: value
                for key, value in asdict(parallel_dims).items()
                if key != "_world_mesh"
            },
            "tp_plan": (
                "TorchTitan apply_non_moe_tp; sequence-sharded residual stream"
                if args.mode == "tp-sp"
                else "Column-parallel QKV/up/gate and row-parallel O/down; replicated residual stream"
            ),
            "compiled_model": args.compile_model,
            "async_tensor_parallel": args.async_tp,
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
