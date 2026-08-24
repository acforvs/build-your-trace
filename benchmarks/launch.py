#!/usr/bin/env python3
"""Launch reproducible public trace-capture workloads without cluster tooling."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"


@dataclass(frozen=True)
class Scenario:
    description: str
    script: str
    processes: int
    arguments: tuple[str, ...]
    requires_torchtitan: bool = True
    supports_perfetto: bool = True
    labeled_arguments: tuple[str, ...] = ("--emit-torch-nvtx", "--module-labels")
    labeled_remove: tuple[str, ...] = ()
    capture_modes: tuple[str, ...] = ("none", "clean", "labeled", "perfetto")


SCENARIOS = {
    "collectives-8": Scenario(
        "Eight-GPU NCCL all-gather and reduce-scatter latency sweep",
        "profile_collectives.py",
        8,
        ("--sizes-mb", "16,64,142,604", "--warmup", "5", "--steps", "20"),
        requires_torchtitan=False,
        supports_perfetto=False,
        labeled_arguments=(),
        capture_modes=("none",),
    ),
    "single-gpu-dense": Scenario(
        "Single-GPU dense forward, backward, and optimizer step",
        "profile_single_gpu_dense.py",
        1,
        ("--batch", "1", "--sequence", "4096", "--warmup", "3", "--steps", "1"),
        requires_torchtitan=False,
        supports_perfetto=False,
        labeled_arguments=(),
        capture_modes=("none", "clean", "labeled"),
    ),
    "moe-single": Scenario(
        "Single-GPU Qwen-shaped grouped-GEMM MoE forward",
        "profile_torchtitan_moe.py",
        1,
        ("--scenario", "single_gpu", "--batch", "1", "--sequence", "4096", "--warmup", "3", "--steps", "3"),
        supports_perfetto=False,
        labeled_arguments=(),
        capture_modes=("none", "clean", "labeled"),
    ),
    "moe-ep4": Scenario(
        "Four-GPU expert-parallel MoE forward",
        "profile_torchtitan_moe.py",
        4,
        ("--scenario", "ep", "--batch", "1", "--sequence", "4096", "--warmup", "3", "--steps", "3"),
        supports_perfetto=False,
        labeled_arguments=(),
        capture_modes=("none", "clean", "labeled"),
    ),
    "dense-ddp-terminal": Scenario(
        "Eight-GPU DDP with one model-sized terminal bucket",
        "profile_torchtitan_dense_ddp.py",
        8,
        ("--policy", "terminal", "--batch", "1", "--sequence", "4096", "--warmup", "3", "--steps", "1"),
    ),
    "dense-ddp-bucketed": Scenario(
        "Eight-GPU DDP with readiness-driven gradient buckets",
        "profile_torchtitan_dense_ddp.py",
        8,
        ("--policy", "overlap", "--batch", "1", "--sequence", "4096", "--warmup", "3", "--steps", "1"),
    ),
    "dense-tp4": Scenario(
        "Four-GPU dense tensor parallelism with replicated residuals",
        "profile_torchtitan_dense_tp.py",
        4,
        ("--mode", "tp", "--compile-model", "--batch", "1", "--sequence", "4096", "--warmup", "2", "--steps", "1"),
        labeled_remove=("--compile-model",),
    ),
    "dense-tp4-sp": Scenario(
        "Four-GPU tensor plus sequence parallelism",
        "profile_torchtitan_dense_tp.py",
        4,
        ("--mode", "tp-sp", "--compile-model", "--batch", "1", "--sequence", "4096", "--warmup", "2", "--steps", "1"),
        labeled_remove=("--compile-model",),
    ),
    "dense-tp4-async": Scenario(
        "Four-GPU async TP micro-pipeline for TP+SP",
        "profile_torchtitan_dense_tp.py",
        4,
        ("--mode", "tp-sp", "--compile-model", "--async-tp", "--batch", "1", "--sequence", "4096", "--warmup", "2", "--steps", "1"),
        labeled_remove=("--compile-model", "--async-tp"),
    ),
    "dense-fsdp8": Scenario(
        "Eight-GPU two-layer dense FSDP2",
        "profile_torchtitan_dense_fsdp.py",
        8,
        ("--batch", "1", "--sequence", "4096", "--warmup", "2", "--steps", "1"),
    ),
    "dense-fsdp8-granular": Scenario(
        "Eight-GPU dense FSDP2 with separate attention and MLP units",
        "profile_torchtitan_dense_fsdp_submodules.py",
        8,
        ("--batch", "1", "--sequence", "4096", "--warmup", "2", "--steps", "1"),
    ),
    "moe-fsdp8-ep1": Scenario(
        "Eight-GPU MoE with separate dense and full-expert FSDP2 units",
        "profile_torchtitan_moe_fsdp.py",
        8,
        ("--batch", "1", "--sequence", "4096", "--warmup", "2", "--steps", "1"),
    ),
    "moe-fsdp8-ep4": Scenario(
        "Eight-GPU expert FSDP2 plus four-way expert parallelism",
        "profile_torchtitan_qwen3.py",
        8,
        ("--scenario", "fsdp_ep", "--ep-degree", "4", "--prefetch-policy", "dependency_ordered_experts_first_backward", "--batch", "1", "--sequence", "4096", "--warmup", "2", "--steps", "1"),
    ),
    "dense-cp4-allgather": Scenario(
        "Four-GPU ring context parallelism with K/V all-gather",
        "profile_torchtitan_dense_cp.py",
        4,
        ("--rotate-method", "allgather", "--batch", "1", "--sequence", "16384", "--warmup", "2", "--steps", "1"),
    ),
    "dense-cp4-alltoall": Scenario(
        "Four-GPU ring context parallelism with cyclic K/V all-to-all",
        "profile_torchtitan_dense_cp.py",
        4,
        ("--rotate-method", "alltoall", "--batch", "1", "--sequence", "16384", "--warmup", "2", "--steps", "1"),
    ),
}


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description="Launch one of the measured Build Your Trace workloads."
    )
    argument_parser.add_argument(
        "scenario", choices=("list", "validate", *SCENARIOS)
    )
    argument_parser.add_argument(
        "--torchtitan-root",
        type=Path,
        default=Path(os.environ["TORCHTITAN_ROOT"])
        if os.environ.get("TORCHTITAN_ROOT")
        else None,
    )
    argument_parser.add_argument("--output-dir", type=Path)
    argument_parser.add_argument(
        "--capture",
        choices=("none", "clean", "labeled", "perfetto"),
        default="none",
    )
    argument_parser.add_argument("--nproc-per-node", type=int)
    argument_parser.add_argument("--master-port", type=int)
    argument_parser.add_argument("--nsys-bin", default="nsys")
    argument_parser.add_argument(
        "--env", action="append", default=[], metavar="NAME=VALUE"
    )
    argument_parser.add_argument("--dry-run", action="store_true")
    argument_parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Arguments passed to the workload; place this option last.",
    )
    return argument_parser


def without_flags(arguments: list[str], removed: tuple[str, ...]) -> list[str]:
    return [argument for argument in arguments if argument not in removed]


def validate_scenarios() -> None:
    errors = []
    for name, scenario in SCENARIOS.items():
        if scenario.processes < 1:
            errors.append(f"{name}: process count must be positive")
        if not (BENCHMARKS / scenario.script).is_file():
            errors.append(f"{name}: missing {scenario.script}")
        for reserved in ("--output", "--capture", "--perfetto-trace-dir"):
            if reserved in scenario.arguments:
                errors.append(f"{name}: launcher owns {reserved}")
        for removed in scenario.labeled_remove:
            if removed not in scenario.arguments:
                errors.append(f"{name}: labeled removal {removed} is absent")
        if "none" not in scenario.capture_modes:
            errors.append(f"{name}: capture_modes must include none")
        if not set(scenario.capture_modes).issubset({"none", "clean", "labeled", "perfetto"}):
            errors.append(f"{name}: invalid capture mode")
        if scenario.supports_perfetto != ("perfetto" in scenario.capture_modes):
            errors.append(f"{name}: supports_perfetto and capture_modes disagree")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Validated {len(SCENARIOS)} public benchmark scenarios.")


def print_scenarios() -> None:
    width = max(map(len, SCENARIOS))
    for name, scenario in SCENARIOS.items():
        print(f"{name:<{width}}  {scenario.processes} GPU  {scenario.description}")


def parse_environment(values: list[str]) -> dict[str, str]:
    environment = os.environ.copy()
    for value in values:
        name, separator, setting = value.partition("=")
        if not separator or not name:
            raise SystemExit(f"Invalid --env value: {value!r}; expected NAME=VALUE")
        environment[name] = setting
    return environment


def workload_command(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    scenario = SCENARIOS[args.scenario]
    if args.output_dir is None:
        raise SystemExit("--output-dir is required for a benchmark run")
    if scenario.requires_torchtitan and args.torchtitan_root is None:
        raise SystemExit("Set TORCHTITAN_ROOT or pass --torchtitan-root")
    if args.capture not in scenario.capture_modes:
        supported = ", ".join(scenario.capture_modes)
        raise SystemExit(
            f"{args.scenario} does not support --capture {args.capture}; "
            f"choose one of: {supported}"
        )
    output_dir = args.output_dir.expanduser().resolve()
    run_name = f"{args.scenario}-{args.capture}"
    metadata_dir = output_dir / f"{run_name}-metadata"
    process_count = args.nproc_per_node or scenario.processes
    if process_count < 1:
        raise SystemExit("--nproc-per-node must be positive")

    arguments = list(scenario.arguments)
    if args.capture == "labeled":
        arguments = without_flags(arguments, scenario.labeled_remove)
        arguments.extend(scenario.labeled_arguments)
    if args.capture in ("clean", "labeled"):
        arguments.append("--capture")
    if args.capture == "perfetto":
        arguments.extend(("--module-labels", "--perfetto-trace-dir", str(output_dir / f"{run_name}-perfetto")))
    if scenario.requires_torchtitan:
        arguments.extend(("--torchtitan-root", str(args.torchtitan_root.expanduser().resolve())))
    arguments.extend(("--output", str(metadata_dir)))
    arguments.extend(args.extra_args)

    torchrun = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={process_count}",
    ]
    if args.master_port is not None:
        torchrun.append(f"--master-port={args.master_port}")
    torchrun.extend((str(BENCHMARKS / scenario.script), *arguments))

    command = torchrun
    if args.capture in ("clean", "labeled"):
        command = [
            args.nsys_bin,
            "profile",
            "--trace=cuda,nvtx,osrt",
            "--sample=none",
            "--cpuctxsw=none",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--force-overwrite=true",
            f"--output={output_dir / run_name}",
            *torchrun,
        ]

    environment = parse_environment(args.env)
    python_paths = [str(BENCHMARKS)]
    if scenario.requires_torchtitan:
        python_paths.insert(0, str(args.torchtitan_root.expanduser().resolve()))
        environment["TORCHTITAN_ROOT"] = str(args.torchtitan_root.expanduser().resolve())
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    environment["PYTHONUNBUFFERED"] = "1"
    return command, environment


def preflight(args: argparse.Namespace) -> None:
    scenario = SCENARIOS[args.scenario]
    if args.dry_run:
        return
    if scenario.requires_torchtitan:
        root = args.torchtitan_root.expanduser().resolve()
        if not (root / "torchtitan").is_dir():
            raise SystemExit(f"TorchTitan package not found under {root}")
    if args.capture in ("clean", "labeled") and shutil.which(args.nsys_bin) is None:
        raise SystemExit(f"Nsight Systems executable not found: {args.nsys_bin}")
    args.output_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parser().parse_args()
    if args.scenario == "list":
        print_scenarios()
        return
    if args.scenario == "validate":
        validate_scenarios()
        return
    command, environment = workload_command(args)
    preflight(args)
    print(shlex.join(command))
    if args.dry_run:
        print(f"PYTHONPATH={environment['PYTHONPATH']}")
        return
    subprocess.run(command, env=environment, check=True)


if __name__ == "__main__":
    main()
