#!/usr/bin/env python3
"""Profile two dense Qwen-shaped layers with attention/MLP FSDP2 units."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
import json
import os
from pathlib import Path
from time import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.models.qwen3.model.model import Qwen3Model

from profile_torchtitan_dense_fsdp import (
    build_model_args,
    parse_args,
    register_dense_ranges,
)
from profile_torchtitan_qwen3 import (
    build_job_config,
    build_parallel_dims,
    describe_prefetch_graph,
    git_revision,
    nvtx_range,
)


def apply_granular_fsdp(
    model: Qwen3Model, parallel_dims, job_config
) -> Qwen3Model:
    dp_mesh = parallel_dims.world_mesh[("dp_shard_cp",)]
    mp_policy = MixedPrecisionPolicy(
        param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
    )
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}

    fully_shard(model.tok_embeddings, **fsdp_config, reshard_after_forward=True)
    for layer in model.layers.values():
        fully_shard(
            [layer.attention_norm, layer.attention],
            **fsdp_config,
            reshard_after_forward=True,
        )
        fully_shard(
            [layer.ffn_norm, layer.feed_forward],
            **fsdp_config,
            reshard_after_forward=True,
        )
    fully_shard(
        [model.norm, model.output],
        **fsdp_config,
        reshard_after_forward=False,
    )
    fully_shard(model, **fsdp_config)
    return model


def configure_granular_prefetch(model: Qwen3Model) -> None:
    layers = list(model.layers.values())
    modules = [
        module
        for layer in layers
        for module in (layer.attention, layer.feed_forward)
    ]
    if not modules:
        return

    model.tok_embeddings.set_modules_to_forward_prefetch([modules[0]])
    for current, following in zip(modules, modules[1:]):
        current.set_modules_to_forward_prefetch([following])
    modules[-1].set_modules_to_forward_prefetch([model.output])

    backward_modules = list(reversed(modules))
    model.output.set_modules_to_backward_prefetch([backward_modules[0]])
    for current, following in zip(backward_modules, backward_modules[1:]):
        current.set_modules_to_backward_prefetch([following])
    backward_modules[-1].set_modules_to_backward_prefetch([model.tok_embeddings])


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    model_args = build_model_args(args)
    parallel_dims = build_parallel_dims("fsdp", world_size, None)
    job_config = build_job_config(args, parallel_dims)

    with torch.device("meta"):
        model = Qwen3Model(model_args)
    model = apply_granular_fsdp(model, parallel_dims, job_config)
    configure_granular_prefetch(model)
    model.to_empty(device=device)
    with torch.no_grad():
        model.init_weights(buffer_device=device)
    model.train()
    if args.module_labels:
        register_dense_ranges(model)
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
        dense_support = script_path.with_name("profile_torchtitan_dense_fsdp.py")
        qwen_support = script_path.with_name("profile_torchtitan_qwen3.py")
        summary = {
            "schema_version": 1,
            "timestamp_unix": time(),
            "scenario": "dense_fsdp_attention_mlp_units",
            "implementation": "TorchTitan dense Qwen3 with granular FSDP2",
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
            "fsdp_units": "attention+attention_norm and feed_forward+ffn_norm",
            "prefetch_policy": "attention_mlp_dependency_ordered",
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
