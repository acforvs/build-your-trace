#!/usr/bin/env python3
"""Profile two Qwen3-235B-A22B-shaped TorchTitan layers on real GPUs."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from contextlib import nullcontext
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
from time import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import FSDPModule

from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.models.moe import MoEArgs
from torchtitan.models.qwen3.infra.parallelize import parallelize_qwen3
from torchtitan.models.qwen3.model.args import Qwen3ModelArgs
from torchtitan.models.qwen3.model.model import Qwen3Model


@contextmanager
def nvtx_range(message: str):
    torch.cuda.nvtx.range_push(message)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def register_module_range(module: torch.nn.Module, name: str) -> None:
    forward_name = f"{name}/forward"
    backward_name = f"{name}/backward"
    forward_records = []
    backward_records = []

    def forward_pre_hook(_module, _inputs, range_name=forward_name):
        torch.cuda.nvtx.range_push(range_name)
        record = torch.profiler.record_function(range_name)
        record.__enter__()
        forward_records.append(record)

    def forward_hook(_module, _inputs, _output):
        forward_records.pop().__exit__(None, None, None)
        torch.cuda.nvtx.range_pop()

    def backward_pre_hook(_module, _grad_output, range_name=backward_name):
        torch.cuda.nvtx.range_push(range_name)
        record = torch.profiler.record_function(range_name)
        record.__enter__()
        backward_records.append(record)

    def backward_hook(_module, _grad_input, _grad_output):
        backward_records.pop().__exit__(None, None, None)
        torch.cuda.nvtx.range_pop()

    module.register_forward_pre_hook(forward_pre_hook)
    module.register_forward_hook(forward_hook)
    module.register_full_backward_pre_hook(backward_pre_hook)
    module.register_full_backward_hook(backward_hook)


def register_layer_ranges(model: Qwen3Model) -> None:
    register_module_range(model.tok_embeddings, "Embedding")
    for layer_id, layer in model.layers.items():
        register_module_range(layer, f"Layer/{layer_id}")
        register_module_range(layer.attention, f"Layer/{layer_id}/attention")
        register_module_range(layer.moe.router, f"Layer/{layer_id}/router")
        register_module_range(layer.moe, f"Layer/{layer_id}/moe")
        register_module_range(layer.moe.experts, f"Layer/{layer_id}/experts")
    register_module_range(model.norm, "FinalNorm")
    register_module_range(model.output, "OutputProjection")


def configure_dependency_ordered_prefetch(
    model: Qwen3Model, *, backward_experts_first: bool = False
) -> None:
    layers = list(model.layers.values())
    if not layers:
        return

    def forward_layer_modules(layer: torch.nn.Module) -> list[torch.nn.Module]:
        modules = [layer]
        if layer.moe_enabled:
            modules.append(layer.moe.experts)
        return modules

    def backward_layer_modules(layer: torch.nn.Module) -> list[torch.nn.Module]:
        if layer.moe_enabled and backward_experts_first:
            return [layer.moe.experts, layer]
        return forward_layer_modules(layer)

    model.tok_embeddings.set_modules_to_forward_prefetch(forward_layer_modules(layers[0]))
    for index, layer in enumerate(layers):
        if index + 1 < len(layers):
            layer.set_modules_to_forward_prefetch(forward_layer_modules(layers[index + 1]))
        else:
            layer.set_modules_to_forward_prefetch([model.output])

    model.output.set_modules_to_backward_prefetch(backward_layer_modules(layers[-1]))
    for reverse_index, layer in enumerate(reversed(layers)):
        if reverse_index + 1 < len(layers):
            previous_layer = layers[-reverse_index - 2]
            layer.set_modules_to_backward_prefetch(backward_layer_modules(previous_layer))
        else:
            layer.set_modules_to_backward_prefetch([model.tok_embeddings])


def describe_prefetch_graph(model: Qwen3Model) -> list[dict[str, object]]:
    names_by_state: dict[int, list[str]] = {}
    state_by_id: dict[int, object] = {}
    for module_name, module in model.named_modules():
        if isinstance(module, FSDPModule):
            state = module._get_fsdp_state()
            state_id = id(state)
            state_by_id[state_id] = state
            names_by_state.setdefault(state_id, []).append(module_name or "<root>")

    def state_names(state: object) -> list[str]:
        return names_by_state.get(id(state), ["<unknown>"])

    graph = []
    for state_id, names in names_by_state.items():
        state = state_by_id[state_id]
        graph.append(
            {
                "module": names,
                "forward": [
                    state_names(prefetch_state)
                    for prefetch_state in state._states_to_forward_prefetch
                ],
                "backward": [
                    state_names(prefetch_state)
                    for prefetch_state in state._states_to_backward_prefetch
                ],
            }
        )
    return graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario", choices=("single_gpu", "fsdp", "fsdp_ep"), required=True
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument(
        "--ep-degree",
        type=int,
        default=None,
        help="Expert-parallel degree for fsdp_ep; defaults to WORLD_SIZE.",
    )
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument(
        "--emit-torch-nvtx",
        action="store_true",
        help="Emit PyTorch record_function ranges, including FSDP module FQNs.",
    )
    parser.add_argument(
        "--module-labels",
        action="store_true",
        help="Register explicit embedding, attention, MoE, norm, and output ranges for diagnostic captures.",
    )
    parser.add_argument(
        "--prefetch-policy",
        choices=(
            "torchtitan",
            "dependency_ordered",
            "dependency_ordered_experts_first_backward",
        ),
        default="dependency_ordered",
        help="Include endpoint expert modules so current-layer experts cannot be overtaken by next-layer prefetch.",
    )
    parser.add_argument(
        "--perfetto-trace-dir",
        type=Path,
        help="Export a labeled rank-0 PyTorch Chrome trace for Perfetto inspection.",
    )
    parser.add_argument(
        "--torchtitan-root",
        type=Path,
        default=Path(os.environ.get("TORCHTITAN_ROOT", ".")),
    )
    return parser.parse_args()


def build_model_args(args: argparse.Namespace) -> Qwen3ModelArgs:
    return Qwen3ModelArgs(
        vocab_size=args.vocab_size,
        max_seq_len=args.sequence,
        head_dim=128,
        dim=4096,
        n_layers=2,
        n_heads=64,
        n_kv_heads=4,
        qk_norm=True,
        hidden_dim=12288,
        rope_theta=5_000_000,
        moe_enabled=True,
        moe_inter_dim=1536,
        moe_args=MoEArgs(
            num_experts=128,
            num_shared_experts=0,
            top_k=8,
            score_func="softmax",
            route_norm=True,
            route_scale=1.0,
            score_before_experts=False,
            use_grouped_mm=True,
            load_balance_coeff=None,
            _debug_force_load_balance=True,
        ),
    )


def build_parallel_dims(
    scenario: str, world_size: int, ep_degree: int | None
) -> ParallelDims:
    if scenario == "single_gpu" and world_size != 1:
        raise ValueError("single_gpu requires WORLD_SIZE=1")
    if scenario != "fsdp_ep" and ep_degree is not None:
        raise ValueError("--ep-degree is only valid with --scenario fsdp_ep")
    resolved_ep_degree = ep_degree or world_size if scenario == "fsdp_ep" else 1
    if world_size % resolved_ep_degree != 0:
        raise ValueError("WORLD_SIZE must be divisible by the expert-parallel degree")
    return ParallelDims(
        dp_replicate=1,
        dp_shard=1 if scenario == "single_gpu" else world_size,
        cp=1,
        tp=1,
        pp=1,
        ep=resolved_ep_degree,
        etp=1,
        world_size=world_size,
    )


def build_job_config(args: argparse.Namespace, parallel_dims: ParallelDims) -> JobConfig:
    config = JobConfig()
    config.training.seq_len = args.sequence
    config.training.local_batch_size = args.batch
    config.training.dtype = "bfloat16"
    config.training.mixed_precision_param = "bfloat16"
    config.training.mixed_precision_reduce = "float32"
    config.training.debug_moe_force_load_balance = True
    config.parallelism.data_parallel_replicate_degree = parallel_dims.dp_replicate
    config.parallelism.data_parallel_shard_degree = parallel_dims.dp_shard
    config.parallelism.tensor_parallel_degree = parallel_dims.tp
    config.parallelism.context_parallel_degree = parallel_dims.cp
    config.parallelism.pipeline_parallel_degree = parallel_dims.pp
    config.parallelism.expert_parallel_degree = parallel_dims.ep
    config.parallelism.expert_tensor_parallel_degree = parallel_dims.etp
    config.parallelism.fsdp_reshard_after_forward = "default"
    config.activation_checkpoint.mode = "none"
    config.compile.enable = False
    config.model.converters = []
    return config


def git_revision(repository: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"



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
    parallel_dims = build_parallel_dims(args.scenario, world_size, args.ep_degree)
    job_config = build_job_config(args, parallel_dims)

    with torch.device("meta"):
        model = Qwen3Model(model_args)
    model = parallelize_qwen3(model, parallel_dims, job_config)
    if args.prefetch_policy != "torchtitan" and args.scenario == "fsdp_ep":
        configure_dependency_ordered_prefetch(
            model,
            backward_experts_first=(
                args.prefetch_policy
                == "dependency_ordered_experts_first_backward"
            ),
        )
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
        raise ValueError("Use separate runs for --emit-torch-nvtx and --perfetto-trace-dir")
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
        profiler.export_chrome_trace(
            str(args.perfetto_trace_dir / "rank-0.json")
        )

    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        script_path = Path(__file__).resolve()
        summary = {
            "schema_version": 1,
            "timestamp_unix": time(),
            "scenario": args.scenario,
            "implementation": "TorchTitan Qwen3 MoE",
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
            "prefetch_policy": args.prefetch_policy,
            "prefetch_graph": prefetch_graph,
            "module_labels": args.module_labels,
            "expert_fsdp_degree": (
                parallel_dims.dp_shard
                * parallel_dims.cp
                * parallel_dims.tp
                // parallel_dims.ep
                if parallel_dims.ep > 1
                else parallel_dims.dp_shard
            ),
            "mean_step_ms": sum(step_times) / len(step_times),
            "step_ms": step_times,
        }
        summary_path = args.output / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
