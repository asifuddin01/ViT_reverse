# CPU inference and analytical cost

## Reproduce

The pinned ViT-Base/16 weights must be cached for the offline command. Run from the project root:

```bash
vit-lab benchmark --config configs/vit_base.yaml \
  --output results/tables/benchmark.csv --warmups 2 --trials 5 \
  --threads 4 --offline
python scripts/plot_benchmark.py results/tables/benchmark.csv \
  results/figures/benchmark_resolution.svg
```

The benchmark creates a fresh Python process for each case, verifies the pinned checkpoint SHA-256, uses CPU FP32, fixes PyTorch to four compute threads, and runs on one seeded random input shape per case. It excludes data loading and preprocessing. Two forward passes warm up each case; five subsequent `torch.inference_mode()` forwards are timed individually with `perf_counter_ns`. Reported latency is the median; throughput is batch size divided by that median. The [JSON record](../results/tables/benchmark.json) keeps every trial time and the hardware/software metadata. The tested machine identified itself as **Apple M1**, arm64, eight logical CPUs, macOS 26.5.2, Python 3.12.13, PyTorch 2.14.1, and timm 1.0.30.

The educational model uses explicit eager attention. The timm reference uses its fused-attention path at 224 pixels. The educational model enables positional-embedding interpolation for 384 and 512 pixels; its output shape is checked after warmup. These timings compare implementations under this CPU setup and are not claims about other devices or inference stacks.

## Measured run

| Implementation | Input | Batch | Median latency | Throughput | Sampled process RSS |
|---|---:|---:|---:|---:|---:|
| Educational manual attention | 224 px | 1 | 81.0 ms | 12.35 images/s | 598 MB |
| Educational manual attention | 224 px | 8 | 889.1 ms | 9.00 images/s | 501 MB |
| Educational manual attention | 384 px | 1 | 268.6 ms | 3.72 images/s | 456 MB |
| Educational manual attention | 512 px | 1 | 1,244.1 ms | 0.80 images/s | 529 MB |
| timm fused attention | 224 px | 1 | 75.9 ms | 13.18 images/s | 408 MB |
| timm fused attention | 224 px | 8 | 536.0 ms | 14.93 images/s | 481 MB |

Each model has **86,567,656 trainable parameters** and **346,270,624 FP32 parameter bytes** (346.3 MB, decimal). The [CSV](../results/tables/benchmark.csv) contains the full numerical values, trial count, device, dtype, hardware, software, kernel, and timing/memory methods. The [resolution plot](../results/figures/benchmark_resolution.svg) shows the three educational batch-1 cases.

The timed 512-pixel educational forwards ranged from **861 to 2,674 ms**, and a preceding exploratory run was materially faster. Other applications were active on this host. These measurements are a local snapshot with visible contention, not a stable estimate or ranking of the two implementations. The batch-8 educational result is especially sensitive to that contention. Repeat on an otherwise idle machine before making performance claims.

The memory column is **process resident set size**, sampled by `psutil` every 5 ms during timed trials and after each forward, following warmup. It includes model weights, Python/PyTorch runtime, allocator state, and activations; sampling can miss brief peaks. Independent fresh processes make cases separate, but memory pressure and allocator behavior still vary. The nonmonotonic values do not imply that higher resolution needs fewer attention bytes. This is not a CUDA/MPS peak-allocation measurement.

## Analytical count and resolution scaling

The code counts convolution, linear, QK, and attention-value matrix multiply-accumulates (MACs). One MAC is reported as two FLOPs. Bias additions, normalization, softmax, GELU, positional interpolation, and memory traffic are excluded, so the FLOP column is a convention-based estimate, not a measured hardware operation count. The formula matched the profiled convolution/matrix FLOPs of a small test ViT.

For resolution `R`, patch size `P=16`, width `D=768`, depth `L=12`, MLP width `M=3072`, and `N=(R/P)^2+1` tokens, the counted terms per image are:

```text
patch projection:      (R/P)^2 · P^2 · C · D
QKV + output project:  L · 4 · N · D^2
QK + attention-value:  L · 2 · N^2 · D
MLP:                   L · 2 · N · D · M
classification head:  D · classes
```

| Resolution | Patches | Tokens with CLS | MACs/image | 2-FLOP estimate | All-layer FP32 attention maps if retained |
|---|---:|---:|---:|---:|---:|
| 224 px | 196 | 197 | 17.56 G | 35.13 G | 22.4 MB |
| 384 px | 576 | 577 | 55.48 G | 110.97 G | 191.8 MB |
| 512 px | 1,024 | 1,025 | 107.03 G | 214.06 G | 605.2 MB |

The final column is an analytical storage size for `12 layers × 12 heads × N² × 4 bytes` for one sample. Normal inference does not retain all those maps. Relative to 224 pixels, that all-layer map size grows about **8.6×** at 384 and **27.0×** at 512. This quadratic attention component is distinct from measured process RSS.
