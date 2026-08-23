#!/usr/bin/env python3
"""Profile terminal and overlapping DDP gradient synchronization on eight GPUs."""

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
from time import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from torch.nn.parallel import DistributedDataParallel
from torchtitan.models.qwen3.model.model import Qwen3Model

from profile_torchtitan_dense_fsdp import build_model_args, register_dense_ranges
from profile_torchtitan_qwen3 import git_revision, nvtx_range


class BucketTraceState:
    def __init__(
        self,
        process_group: dist.ProcessGroup,
        world_size: int,
        parameter_names: dict[int, str],
    ) -> None:
        self.process_group = process_group
        self.world_size = world_size
        self.parameter_names = parameter_names
        self.capture = False
        self.buckets: list[dict[str, int | bool]] = []


def allreduce_with_metadata(
    state: BucketTraceState, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    tensor = bucket.buffer()
    if state.capture:
        state.buckets.append(
            {
                "launch_order": len(state.buckets),
                "bucket_index": bucket.index(),
                "elements": tensor.numel(),
                "bytes": tensor.numel() * tensor.element_size(),
                "is_last": bucket.is_last(),
                "parameters": [
                    state.parameter_names.get(id(parameter), "<unknown>")
                    for parameter in bucket.parameters()
                ],
            }
        )
    with nvtx_range(f"DDP/bucket_{bucket.index()}/all_reduce_launch"):
        return default_hooks.allreduce_hook(state.process_group, bucket)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("terminal", "overlap"), required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260823)
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


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 8:
        raise ValueError("This benchmark requires exactly eight GPUs")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    model_args = build_model_args(args)
    with torch.device("meta"):
        model = Qwen3Model(model_args)
    model.to_empty(device=device)
    with torch.no_grad():
        model.init_weights(buffer_device=device)
    model.to(dtype=torch.bfloat16)
    model.train()
    if args.module_labels:
        register_dense_ranges(model)

    bucket_cap_mb = 2048 if args.policy == "terminal" else 96
    parameter_names = {
        id(parameter): name for name, parameter in model.named_parameters()
    }
    model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        bucket_cap_mb=bucket_cap_mb,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
    )
    bucket_state = BucketTraceState(dist.group.WORLD, world_size, parameter_names)
    model.register_comm_hook(bucket_state, allreduce_with_metadata)

    tokens = torch.arange(
        args.batch * args.sequence, device=device, dtype=torch.long
    ).reshape(args.batch, args.sequence) % args.vocab_size
    targets = torch.roll(tokens, shifts=-1, dims=1)

    def train_step(record_buckets: bool = False) -> float:
        model.zero_grad(set_to_none=True)
        bucket_state.capture = record_buckets
        if record_buckets:
            bucket_state.buckets = []
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
        torch.cuda.synchronize(device)
        end.record()
        end.synchronize()
        bucket_state.capture = False
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
        step_times = [train_step(record_buckets=True) for _ in range(args.steps)]
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
            "scenario": "dense_ddp",
            "policy": args.policy,
            "implementation": "PyTorch DDP over two dense Qwen3-shaped layers",
            "torchtitan_revision": git_revision(args.torchtitan_root),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "nccl_version": ".".join(map(str, torch.cuda.nccl.version())),
            "gpu": torch.cuda.get_device_name(device),
            "world_size": world_size,
            "bucket_cap_mb": bucket_cap_mb,
            "broadcast_buffers": False,
            "gradient_as_bucket_view": True,
            "model": {
                "shape_reference": "Qwen3-235B-A22B attention width",
                "layers": 2,
                "vocab_size": args.vocab_size,
                "dim": model_args.dim,
                "heads": model_args.n_heads,
                "kv_heads": model_args.n_kv_heads,
                "head_dim": model_args.head_dim,
                "dense_intermediate_dim": model_args.hidden_dim,
                "parameter_dtype": "BF16",
            },
            "workload": {
                "batch_per_rank": args.batch,
                "sequence": args.sequence,
                "synthetic_tokens": True,
                "loss": "cross_entropy",
                "optimizer": "excluded",
            },
            "module_labels": args.module_labels,
            "mean_step_ms": sum(step_times) / len(step_times),
            "step_ms": step_times,
            "bucket_launches": bucket_state.buckets,
            "total_gradient_bytes": sum(
                int(bucket["bytes"]) for bucket in bucket_state.buckets
            ),
        }
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print(json.dumps(summary, indent=2))

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
