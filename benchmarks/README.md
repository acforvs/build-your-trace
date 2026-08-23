# Reproducing the profiling workloads

These scripts run synthetic Qwen-shaped models with real PyTorch kernels, autograd, and distributed collectives. They need no dataset, tokenizer, or checkpoint. They are calibration workloads rather than production training recipes, so record the GPU, topology, driver, CUDA, NCCL, PyTorch, and TorchTitan versions with every capture.

## Environment

Use a disposable container or virtual environment. Do not upgrade PyTorch inside a shared training environment.

The TorchTitan workloads target the public `v0.2.0` tag:

```bash
git clone --branch v0.2.0 --depth 1 https://github.com/pytorch/torchtitan.git /path/to/torchtitan
export TORCHTITAN_ROOT=/path/to/torchtitan
export TRACE_RESULTS=/path/to/trace-results
```

Install a PyTorch build compatible with your NVIDIA driver and follow TorchTitan's installation notes in that isolated environment. Nsight Systems is required only for `clean` and `labeled` captures.

The full MoE FSDP scenarios use TorchTitan's debug balanced round-robin routing override. This gives every expert the same number of synthetic token copies, making expert compute and all-to-all sizes deterministic; these runs do not measure natural router imbalance. The standalone `moe-single` and `moe-ep4` scenarios use ordinary softmax top-k routing.

The CP scenarios are custom benchmark integrations: they apply PyTorch's native `context_parallel` context to the TorchTitan v0.2.0 Qwen3 model and explicitly shard its RoPE cache. They do not imply that TorchTitan v0.2.0 shipped supported Qwen3 CP integration.

## Launcher

```bash
python3 benchmarks/launch.py validate
python3 benchmarks/launch.py list
```

Preview a command without importing PyTorch or touching a GPU:

```bash
python3 benchmarks/launch.py dense-fsdp8 \
  --torchtitan-root "$TORCHTITAN_ROOT" \
  --output-dir "$TRACE_RESULTS" \
  --capture clean \
  --dry-run
```

Record clean timing and labeled correlation in separate runs:

```bash
python3 benchmarks/launch.py moe-fsdp8-ep4 --torchtitan-root "$TORCHTITAN_ROOT" --output-dir "$TRACE_RESULTS" --capture clean
python3 benchmarks/launch.py moe-fsdp8-ep4 --torchtitan-root "$TORCHTITAN_ROOT" --output-dir "$TRACE_RESULTS" --capture labeled
```

For TP workloads, clean captures keep compilation enabled while labeled captures use eager execution to preserve useful module boundaries. Async TP uses synchronous TP+SP only for the label-correlation run; its clean capture remains the timing source.

Some scenarios can also export a PyTorch/Perfetto trace:

```bash
python3 benchmarks/launch.py dense-cp4-allgather --torchtitan-root "$TORCHTITAN_ROOT" --output-dir "$TRACE_RESULTS" --capture perfetto
```

The launcher is single-node. It does not claim to measure multi-node fabric behavior. Store all profiler output outside this repository: reports can contain hostnames, local paths, device identifiers, and process environments.
