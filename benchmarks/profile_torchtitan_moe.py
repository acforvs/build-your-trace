#!/usr/bin/env python3
"""Profile one Qwen3-235B-A22B-shaped MoE forward path with TorchTitan."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
from time import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed._functional_collectives import (
    all_to_all_single,
    all_to_all_single_autograd,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module

from torchtitan.distributed.expert_parallel import ExpertParallel
from torchtitan.models.moe import MoEArgs
from torchtitan.models.moe.moe import MoE
from torchtitan.models.moe.utils import _permute, _unpermute


@contextmanager
def nvtx_range(message: str):
    torch.cuda.nvtx.range_push(message)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


class ProfiledExpertParallel(ExpertParallel):
    """TorchTitan ExpertParallel with ranges at its semantic boundaries."""

    def _token_dispatch(self, mod, inputs, device_mesh):
        routed_input, num_tokens_per_expert = inputs
        ep_degree = device_mesh.shape[0]
        num_local_experts = num_tokens_per_expert.shape[0] // ep_degree

        with nvtx_range("EP/dispatch metadata"):
            with torch.no_grad():
                num_tokens_per_expert_group = all_to_all_single(
                    num_tokens_per_expert,
                    None,
                    None,
                    group=device_mesh.get_group(),
                )
                num_tokens_per_expert_group = torch.ops._c10d_functional.wait_tensor(
                    num_tokens_per_expert_group
                )
                input_splits = (
                    num_tokens_per_expert.view(ep_degree, -1)
                    .sum(dim=1)
                    .to(torch.device("cpu"), non_blocking=True)
                )
                output_splits = (
                    num_tokens_per_expert_group.view(ep_degree, -1)
                    .sum(dim=1)
                    .to(torch.device("cpu"), non_blocking=False)
                )
                self.input_splits = input_splits.tolist()
                self.output_splits = output_splits.tolist()

        with nvtx_range("EP/dispatch all-to-all"):
            routed_input = all_to_all_single_autograd(
                routed_input,
                self.output_splits,
                self.input_splits,
                device_mesh.get_group(),
            )

        with nvtx_range("EP/destination permute"):
            (
                self.input_shape,
                routed_input,
                self.permuted_indices,
                num_tokens_per_expert_group,
            ) = _permute(
                routed_input,
                num_tokens_per_expert_group,
                ep_degree,
                num_local_experts,
            )

        return routed_input, num_tokens_per_expert_group

    def _token_combine(self, mod, routed_output, device_mesh):
        with nvtx_range("EP/destination unpermute"):
            routed_output = _unpermute(
                routed_output, self.input_shape, self.permuted_indices
            )

        with nvtx_range("EP/combine all-to-all"):
            routed_output = all_to_all_single_autograd(
                routed_output,
                self.input_splits,
                self.output_splits,
                device_mesh.get_group(),
            )
        return routed_output


class ProfiledMoE(MoE):
    """TorchTitan MoE forward with ranges matching the teaching blocks."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, dim = x.shape
        x = x.view(-1, dim)

        with nvtx_range("MoE/router GEMM"):
            scores = self.router.gate(x)

        with nvtx_range("MoE/softmax and top-k"):
            if self.router.score_func == "sigmoid":
                scores = torch.sigmoid(scores.to(torch.float32))
            elif self.router.score_func == "softmax":
                scores = F.softmax(scores.to(torch.float32), dim=1)
            else:
                raise NotImplementedError(
                    f"Unknown score function {self.router.score_func}"
                )
            top_scores, selected_experts_indices = torch.topk(
                scores, k=self.router.top_k, dim=1
            )
            if self.router.route_norm:
                denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
                top_scores = top_scores / denominator
            top_scores = top_scores * self.router.route_scale
            num_tokens_per_expert = torch.histc(
                selected_experts_indices.view(-1),
                bins=self.router.num_experts,
                min=0,
                max=self.router.num_experts,
            )

        with torch.no_grad():
            self.tokens_per_expert.add_(num_tokens_per_expert)

        with nvtx_range("MoE/source token permute"):
            (
                top_scores_experts_sorted,
                token_indices_experts_sorted,
                num_tokens_per_expert,
            ) = self.reorderer(top_scores, selected_experts_indices)
            token_indices_experts_sorted = token_indices_experts_sorted.reshape(
                -1, 1
            ).expand(-1, dim)
            routed_input = torch.gather(
                x, dim=0, index=token_indices_experts_sorted
            )

        with nvtx_range("MoE/grouped SwiGLU experts"):
            routed_output = self.experts(routed_input, num_tokens_per_expert)

        with nvtx_range("MoE/source token unpermute"):
            routed_output = (
                routed_output.to(torch.float32)
                * top_scores_experts_sorted.reshape(-1, 1)
            ).to(x.dtype)
            out = torch.zeros_like(x)
            out = out.scatter_add(
                dim=0, index=token_indices_experts_sorted, src=routed_output
            )

        return out.reshape(batch_size, sequence_length, dim)


@dataclass(frozen=True)
class ModelShape:
    hidden_size: int = 4096
    expert_intermediate_size: int = 1536
    experts: int = 128
    active_experts: int = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("single_gpu", "ep"), required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument(
        "--torchtitan-root",
        type=Path,
        default=Path(os.environ.get("TORCHTITAN_ROOT", ".")),
    )
    return parser.parse_args()



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

    if args.scenario == "single_gpu" and world_size != 1:
        raise ValueError("single_gpu requires WORLD_SIZE=1")
    if args.scenario == "ep" and world_size <= 1:
        raise ValueError("ep requires WORLD_SIZE>1")

    shape = ModelShape()
    moe_args = MoEArgs(
        num_experts=shape.experts,
        num_shared_experts=0,
        top_k=shape.active_experts,
        score_func="softmax",
        route_norm=True,
        route_scale=1.0,
        score_before_experts=False,
        use_grouped_mm=True,
        load_balance_coeff=None,
        _debug_force_load_balance=False,
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    with torch.device("meta"):
        moe = ProfiledMoE(moe_args, shape.hidden_size, shape.expert_intermediate_size)

    if args.scenario == "ep":
        ep_mesh = init_device_mesh(
            "cuda", (world_size,), mesh_dim_names=("ep",)
        )
        moe.experts = parallelize_module(
            moe.experts, ep_mesh, ProfiledExpertParallel()
        )

    moe.to_empty(device=device)
    moe.to(dtype=torch.bfloat16)
    with torch.no_grad():
        moe.init_weights(init_std=0.02, buffer_device=device)
    moe.eval()

    input_generator = torch.Generator(device=device)
    input_generator.manual_seed(args.seed + rank)
    hidden_states = torch.randn(
        args.batch,
        args.sequence,
        shape.hidden_size,
        dtype=torch.bfloat16,
        device=device,
        generator=input_generator,
    )

    def forward_step() -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad(), nvtx_range("MoE/forward"):
            output = moe(hidden_states)
        output.record_stream(torch.cuda.current_stream())
        end.record()
        end.synchronize()
        return start.elapsed_time(end)

    for _ in range(args.warmup):
        forward_step()
    dist.barrier()

    if args.capture:
        torch.cuda.cudart().cudaProfilerStart()
    step_times = [forward_step() for _ in range(args.steps)]
    torch.cuda.synchronize(device)
    if args.capture:
        torch.cuda.cudart().cudaProfilerStop()
    dist.barrier()

    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        script_path = Path(__file__).resolve()
        summary = {
            "schema_version": 1,
            "timestamp_unix": time(),
            "scenario": args.scenario,
            "implementation": "TorchTitan Qwen3 MoE grouped GEMM",
            "torchtitan_revision": git_revision(args.torchtitan_root),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "nccl_version": ".".join(map(str, torch.cuda.nccl.version())),
            "gpu": torch.cuda.get_device_name(device),
            "world_size": world_size,
            "model": asdict(shape),
            "workload": {
                "batch_per_rank": args.batch,
                "sequence": args.sequence,
                "synthetic_hidden_states": True,
                "forward_only": True,
                "parameter_dtype": "BF16",
            },
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
