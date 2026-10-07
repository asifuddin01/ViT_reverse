"""Fresh-process CPU benchmarks for the educational ViT and pinned timm model."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import psutil
import timm
import torch
import torchvision
import yaml
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from vit_lab.models.vision_transformer import VisionTransformer
from vit_lab.reference.loader import load_reference_model, sha256_file


def _cpu_model() -> str:
    if sys.platform == "darwin":
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return platform.processor() or platform.machine()


def analytical_macs(model: dict[str, Any], resolution: int) -> dict[str, int]:
    """Count conv/linear/attention matrix MACs for one image, excluding elementwise ops."""
    patch = int(model["patch_size"])
    if resolution <= 0 or resolution % patch:
        raise ValueError("Resolution must be positive and divisible by patch size")
    patches = (resolution // patch) ** 2
    tokens = patches + 1
    width = int(model["embed_dim"])
    depth = int(model["depth"])
    hidden = int(width * float(model["mlp_ratio"]))
    channels = int(model["in_channels"])
    classes = int(model["num_classes"])
    patch_macs = patches * patch * patch * channels * width
    qkv_and_projection = depth * (4 * tokens * width * width)
    attention = depth * (2 * tokens * tokens * width)
    mlp = depth * (2 * tokens * width * hidden)
    head = width * classes
    total = patch_macs + qkv_and_projection + attention + mlp + head
    return {
        "patches": patches,
        "tokens": tokens,
        "patch_macs": patch_macs,
        "qkv_and_projection_macs": qkv_and_projection,
        "attention_macs": attention,
        "mlp_macs": mlp,
        "head_macs": head,
        "macs_per_image": total,
        "flops_2_per_mac": 2 * total,
        "all_layer_attention_map_bytes_fp32": depth * int(model["num_heads"]) * tokens**2 * 4,
    }


def _load_model(config: dict[str, Any], implementation: str, offline: bool):
    reference = config["reference"]
    if implementation == "timm_reference":
        model, _hub_config, checksum = load_reference_model(
            reference, local_files_only=offline
        )
        return model, checksum
    if implementation != "educational":
        raise ValueError(f"Unknown implementation: {implementation}")
    checkpoint = hf_hub_download(
        repo_id=reference["repo_id"], revision=reference["revision"],
        filename="model.safetensors", local_files_only=offline,
    )
    checksum = sha256_file(checkpoint)
    if checksum != reference["checkpoint_sha256"]:
        raise ValueError("Pinned checkpoint SHA-256 mismatch")
    model = VisionTransformer(**config["model"], dynamic_image_size=True)
    state = load_file(checkpoint)
    model.load_state_dict(state, strict=True)
    del state
    gc.collect()
    return model, checksum


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    location = (len(ordered) - 1) * percentile
    lower = int(location)
    fraction = location - lower
    return ordered[lower] * (1 - fraction) + ordered[min(lower + 1, len(ordered) - 1)] * fraction


def measure_case(payload: dict[str, Any]) -> dict[str, Any]:
    """Measure one case in a fresh process so RSS samples are case-local."""
    config = payload["config"]
    implementation = payload["implementation"]
    resolution = int(payload["resolution"])
    batch_size = int(payload["batch_size"])
    warmups = int(payload["warmups"])
    trials = int(payload["trials"])
    threads = int(payload["threads"])
    if batch_size <= 0 or warmups < 1 or trials < 2 or threads <= 0:
        raise ValueError("Need positive batch/threads, at least one warmup, and two trials")
    if implementation == "timm_reference" and resolution != config["model"]["image_size"]:
        raise ValueError("The pinned timm reference uses its fixed 224-pixel input size")
    counts = analytical_macs(config["model"], resolution)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(7)
    model, checksum = _load_model(config, implementation, bool(payload["offline"]))
    model.eval()
    attention_kernel = (
        "timm_fused" if model.blocks[0].attn.fused_attn else "timm_manual"
    ) if implementation == "timm_reference" else "educational_manual"
    input_tensor = torch.randn(batch_size, int(config["model"]["in_channels"]),
                               resolution, resolution)
    with torch.inference_mode():
        for _ in range(warmups):
            output = model(input_tensor)
            if output.shape != (batch_size, config["model"]["num_classes"]):
                raise RuntimeError("Unexpected benchmark output shape")
    process = psutil.Process()
    baseline_rss = process.memory_info().rss
    peak_rss = baseline_rss
    stop = threading.Event()

    def sample_memory() -> None:
        nonlocal peak_rss
        while not stop.is_set():
            peak_rss = max(peak_rss, process.memory_info().rss)
            stop.wait(0.005)

    sampler = threading.Thread(target=sample_memory, daemon=True)
    samples_ms = []
    sampler.start()
    try:
        with torch.inference_mode():
            for _ in range(trials):
                start = time.perf_counter_ns()
                output = model(input_tensor)
                elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
                samples_ms.append(elapsed_ms)
                peak_rss = max(peak_rss, process.memory_info().rss)
    finally:
        stop.set()
        sampler.join()
    median_ms = statistics.median(samples_ms)
    return {
        "implementation": implementation,
        "device": "cpu",
        "cpu_model": _cpu_model(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "timm_version": timm.__version__,
        "timing_method": "perf_counter_ns; inference_mode; after warmup",
        "memory_method": "psutil sampled RSS during trials (5 ms poll)",
        "attention_kernel": attention_kernel,
        "dtype": "float32",
        "batch_size": batch_size,
        "resolution": resolution,
        "patch_size": config["model"]["patch_size"],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters()
                                         if parameter.requires_grad),
        "fp32_parameter_bytes": sum(parameter.numel() * parameter.element_size()
                                    for parameter in model.parameters()),
        **counts,
        "warmups": warmups,
        "trials": trials,
        "threads": threads,
        "median_latency_ms": median_ms,
        "minimum_latency_ms": min(samples_ms),
        "p90_latency_ms": _percentile(samples_ms, 0.9),
        "throughput_images_per_second": batch_size * 1000 / median_ms,
        "rss_after_warmup_bytes": baseline_rss,
        "sampled_peak_rss_bytes": peak_rss,
        "sampled_extra_rss_bytes": max(0, peak_rss - baseline_rss),
        "checkpoint_sha256": checksum,
        "trial_latencies_ms": samples_ms,
    }


def run_benchmarks(
    config: dict[str, Any], *, output: Path, offline: bool = False,
    warmups: int = 2, trials: int = 5, threads: int = 4,
) -> list[dict[str, Any]]:
    """Run a fixed case matrix and save CSV plus full timing samples/metadata."""
    cases = [
        ("educational", 224, 1), ("educational", 224, 8),
        ("educational", 384, 1), ("educational", 512, 1),
        ("timm_reference", 224, 1), ("timm_reference", 224, 8),
    ]
    rows = []
    for implementation, resolution, batch_size in cases:
        payload = {
            "config": config, "implementation": implementation,
            "resolution": resolution, "batch_size": batch_size,
            "warmups": warmups, "trials": trials, "threads": threads,
            "offline": offline,
        }
        completed = subprocess.run(
            [sys.executable, "-m", "vit_lab.benchmarking.runner", "--worker",
             json.dumps(payload)], capture_output=True, text=True, check=False,
        )
        if completed.returncode:
            raise RuntimeError(
                f"Benchmark {implementation} {resolution}px batch {batch_size} failed:\n"
                f"{completed.stderr}"
            )
        row = json.loads(completed.stdout)
        rows.append(row)
        print(f"{implementation} {resolution}px batch {batch_size}: "
              f"{row['median_latency_ms']:.2f} ms; "
              f"{row['throughput_images_per_second']:.2f} images/s", flush=True)
    metadata = {
        "method": "torch.inference_mode, perf_counter_ns around forward after warmup",
        "memory_method": "psutil process RSS sampled every 5 ms during trials plus after each forward; "
                         "peak is sampled, not an exact device allocation peak",
        "process_isolation": "each case runs in a fresh Python process",
        "mac_convention": "one multiply-accumulate is one MAC and two FLOPs; "
                          "conv, linear, and QK/AV matrix products only; "
                          "bias, normalization, softmax, GELU, and interpolation excluded",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "config": config,
        "cases": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_rows = [{key: value for key, value in row.items() if key != "trial_latencies_ms"}
                for row in rows]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_rows)
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", help="Internal JSON payload for one isolated case")
    args = parser.parse_args()
    if args.worker is None:
        parser.error("Use vit-lab benchmark for the full case matrix")
    print(json.dumps(measure_case(json.loads(args.worker))))


if __name__ == "__main__":
    main()
