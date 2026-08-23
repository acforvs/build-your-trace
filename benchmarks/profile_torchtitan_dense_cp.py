#!/usr/bin/env python3
"""Apply PyTorch native context parallelism to two TorchTitan Qwen-shaped layers."""

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
from torch.nn.attention import SDPBackend
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.utils import create_context_parallel_ctx
from torchtitan.models.attention import ScaledDotProductAttentionWrapper
from torchtitan.models.qwen3.model.args import Qwen3ModelArgs
from torchtitan.models.qwen3.model.model import Qwen3Model

from profile_torchtitan_dense_fsdp import build_model_args, register_dense_ranges
from profile_torchtitan_qwen3 import build_job_config, git_revision, nvtx_range


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--sequence", type=int, default=32768)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument(
        "--rotate-method", choices=("allgather", "alltoall"), required=True
    )
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


def build_cp_dims(world_size: int) -> ParallelDims:
    return ParallelDims(
        dp_replicate=1,
        dp_shard=1,
        cp=world_size,
        tp=1,
        pp=1,
        ep=1,
        etp=1,
        world_size=world_size,
    )


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)

    if args.sequence % (2 * world_size) != 0:
        raise ValueError("Sequence length must be divisible by 2 × CP degree")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    model_args: Qwen3ModelArgs = build_model_args(args)
    parallel_dims = build_cp_dims(world_size)
    job_config = build_job_config(args, parallel_dims)
    job_config.parallelism.context_parallel_rotate_method = args.rotate_method
    ScaledDotProductAttentionWrapper.sdpa_backends = [SDPBackend.FLASH_ATTENTION]

    with torch.device("meta"):
        model = Qwen3Model(model_args)
    model.to_empty(device=device)
    model.to(dtype=torch.bfloat16)
    with torch.no_grad():
        model.init_weights(buffer_device=device)
    model.train()
    if args.module_labels:
        register_dense_ranges(model)

    base_tokens = torch.arange(
        args.batch * args.sequence, device=device, dtype=torch.long
    ).reshape(args.batch, args.sequence) % args.vocab_size
    base_targets = torch.roll(base_tokens, shifts=-1, dims=1)
    cp_mesh = parallel_dims.world_mesh["cp"]

    def train_step() -> float:
        model.zero_grad(set_to_none=True)
        tokens = base_tokens.clone()
        targets = base_targets.clone()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        cp_context = create_context_parallel_ctx(
            cp_mesh=cp_mesh,
            cp_buffers=[tokens, targets, model.rope_cache],
            cp_seq_dims=[1, 1, 0],
            cp_no_restore_buffers={tokens, targets},
            cp_rotate_method=args.rotate_method,
        )
        with cp_context:
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
            "scenario": f"dense_cp_{args.rotate_method}",
            "implementation": "PyTorch native context_parallel context applied by this benchmark to TorchTitan Qwen3",
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
                "sequence_per_rank": args.sequence // world_size,
                "synthetic_tokens": True,
                "loss": "local cross_entropy on the CP sequence shard",
            },
            "parallel_dims": {
                key: value
                for key, value in asdict(parallel_dims).items()
                if key != "_world_mesh"
            },
            "rotate_method": args.rotate_method,
            "sdpa_backend": "FLASH_ATTENTION",
            "algorithm": "ring attention with partial SDPA accumulation",
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
