# Product Requirements Document
## ZeroVRAM-AirLLM-Engine (EX05)

**Version:** 1.0.0
**Status:** Draft
**Author:** Avi Ayeli
**Date:** 2026-06-30

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Goals and Objectives](#3-goals-and-objectives)
4. [System Requirements](#4-system-requirements)
5. [Success Metrics](#5-success-metrics)
6. [Out of Scope](#6-out-of-scope)
7. [Risks and Mitigations](#7-risks-and-mitigations)

---

## 1. Project Overview

**ZeroVRAM-AirLLM-Engine** is a local LLM inference engine designed to run large language models on hardware with severely constrained GPU VRAM. The engine targets consumer-grade integrated graphics environments — specifically Intel Iris Xe or equivalent iGPUs with as little as 128 MB of dedicated VRAM — by offloading model weight storage and paging to system RAM and NVMe/SSD disk via the `airllm` library.

The project demonstrates that useful, interactive LLM inference is achievable without a discrete GPU, using only commodity hardware that is widely available on student laptops and low-cost workstations. This is accomplished through layer-wise model loading (paging individual transformer layers on demand from disk to RAM), combined with aggressive model quantization to reduce per-layer memory footprints.

**Project Classification:** EX05 — Applied Systems Engineering / Memory-Bound Inference

---

## 2. Problem Statement

### 2.1 The Zero-VRAM Bottleneck

Modern large language models range from 7B to 70B+ parameters. Even at 4-bit quantization, a 7B-parameter model requires approximately 3.5–4 GB of weight storage. Consumer discrete GPUs typically carry 6–12 GB VRAM; integrated GPUs share a small window of system memory (64–512 MB) that is wholly insufficient for holding any LLM layer set in VRAM simultaneously.

The conventional GPU-resident inference pipeline (all weights loaded into VRAM before generation) fails immediately on such hardware with an `OutOfMemoryError` at model load time, before a single token is produced.

### 2.2 Why System RAM Alone Is Insufficient for Standard Inference Frameworks

Frameworks such as `llama.cpp` with CPU-only mode and `transformers` with `device_map="cpu"` load the full model into system RAM. On machines with 16 GB RAM, a 7B model at fp16 (14 GB) saturates available memory and leaves no headroom for the OS, tokenizer buffers, or KV-cache — resulting in OOM kills or extreme swap thrashing.

### 2.3 The Gap This Project Fills

There is no production-ready, low-ceremony solution that:

- Operates within an ~8–12 GB RAM working set for a 7B-class model.
- Pages model weights on-demand from disk rather than pre-loading all layers.
- Runs in a standard WSL2/Ubuntu environment without CUDA drivers or a discrete GPU.
- Requires no custom kernel modules, ROCm, or driver installations.

`airllm` provides the layer-paging primitive; this project wraps it into a reproducible, benchmarked, and documented inference pipeline that can serve as a reference implementation for memory-bound deployments.

---

## 3. Goals and Objectives

### 3.1 Primary Goal

Achieve stable, crash-free autoregressive text generation with a 7B-class LLM on a machine with ≤ 128 MB VRAM and 16 GB system RAM, running under WSL2/Ubuntu.

### 3.2 Objectives

| # | Objective | Priority |
|---|-----------|----------|
| O1 | Successfully load and run a 7B-parameter model (e.g., `meta-llama/Llama-2-7b-hf` or equivalent) via `airllm` without OOM crash | P0 |
| O2 | Apply INT4 / GPTQ quantization to reduce per-layer VRAM and RAM peak usage | P0 |
| O3 | Establish a baseline latency measurement (tokens/sec) for the layer-paged inference path | P1 |
| O4 | Compare baseline (naive CPU `transformers`) against AirLLM-paged inference in terms of peak RAM usage and crash rate | P1 |
| O5 | Produce a reproducible environment (pinned `pyproject.toml`, `uv` lock file) that a new contributor can set up in under 10 minutes | P1 |
| O6 | Document the inference pipeline with enough detail to serve as an EX05 academic deliverable | P2 |

### 3.3 Baseline vs. Optimized Comparison

| Dimension | Baseline (naive `transformers` CPU) | AirLLM Optimized |
|-----------|--------------------------------------|------------------|
| Model load strategy | Full model loaded into RAM at startup | Layer-by-layer paging from disk → RAM → (optional) GPU |
| Peak RAM at load | ~14 GB (fp16 7B) | ≤ 8 GB (INT4 7B, single layer in RAM) |
| VRAM required | 0 MB (CPU mode) or full model | 0–128 MB (single layer or skip GPU entirely) |
| OOM crash rate on target HW | High (near 100% for 7B fp16) | Target: 0% |
| First-token latency | N/A (crashes before inference) | Measurable; expected ~10–60 s/token on disk-paged path |
| Tokens/sec (sustained) | N/A | Measurable; tracked per run |

---

## 4. System Requirements

### 4.1 Host Environment

| Requirement | Specification |
|-------------|---------------|
| Host OS | Windows 11 with WSL2 enabled |
| Guest OS | Ubuntu 22.04 LTS (WSL2 distro) |
| CPU | x86-64 (Intel Core i5/i7 or equivalent); AVX2 support preferred |
| System RAM | 16 GB minimum |
| Disk | ≥ 30 GB free NVMe/SSD space (model weights + quantized cache) |
| GPU (optional) | Intel Iris Xe iGPU; ≤ 128 MB VRAM shared; treated as zero-VRAM for weight storage |
| Internet | Required for initial model download from HuggingFace Hub |

### 4.2 Software Dependencies

| Component | Version / Constraint | Notes |
|-----------|----------------------|-------|
| Python | ≥ 3.10, < 3.13 | Managed via `uv`; `.python-version` pins exact release |
| `uv` | Latest stable | Used as the sole package and virtualenv manager; no `pip` or `conda` |
| `airllm` | ≥ 0.8.0 | Core layer-paging inference library |
| `transformers` | ≥ 4.40.0 | Tokenizer and model config support |
| `torch` | ≥ 2.2.0, CPU build | CPU-only PyTorch; CUDA build optional if CUDA is present |
| `accelerate` | ≥ 0.29.0 | Required by `airllm` for device map utilities |
| `bitsandbytes` | ≥ 0.43.0 | INT4/INT8 quantization backend |
| `huggingface-hub` | ≥ 0.22.0 | Authenticated model download |

### 4.3 Project Tooling

- **Package manager:** `uv` only. `pip install` commands are disallowed in documentation and scripts.
- **Lock file:** `uv.lock` is committed and must remain in sync with `pyproject.toml`.
- **Virtual environment:** Created and managed by `uv venv`; never committed to version control.
- **Environment variables:** `HF_TOKEN` must be set in `.env` (gitignored) for gated model access.

### 4.4 Directory Structure (Target)

```
ZeroVRAM-AirLLM-Engine/
├── docs/
│   ├── PRD.md           # This document
│   ├── PLAN.md          # Implementation plan
│   └── TODO.md          # Tracked task list
├── src/
│   └── engine/
│       ├── __init__.py
│       ├── loader.py    # AirLLM model loader wrapper
│       ├── infer.py     # Inference loop and token generation
│       └── benchmark.py # Latency and memory measurement utilities
├── scripts/
│   └── run_inference.py # CLI entrypoint
├── pyproject.toml
├── uv.lock
└── .env.example
```

---

## 5. Success Metrics

### 5.1 Stability (P0)

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| OOM crash rate | 0 out of 5 consecutive inference runs | Manual test: run `scripts/run_inference.py` 5 times with the same prompt; record any `RuntimeError: CUDA out of memory` or `MemoryError` |
| Process exit code | 0 for all successful runs | Shell: `echo $?` after each run |

### 5.2 Latency (P1)

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Time-to-first-token (TTFT) | Recorded per run (no hard cap; characterization only) | `benchmark.py`: `time.perf_counter()` delta from generation call to first decoded token |
| Sustained throughput | Recorded per run in tokens/sec | `benchmark.py`: `(output_tokens - 1) / (end_time - first_token_time)` |
| Peak RSS (Resident Set Size) | ≤ 12 GB during inference | `/proc/self/status` VmRSS sampled at 1 Hz during run via `benchmark.py` |

### 5.3 Reproducibility (P1)

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Setup time (clean environment) | ≤ 10 minutes from `git clone` to first successful inference | Timed walkthrough on a fresh WSL2 instance |
| Dependency resolution failures | 0 | `uv sync` must exit 0 with the committed `uv.lock` |

### 5.4 Acceptance Criteria Summary

A run is considered **passing** if and only if:

1. `scripts/run_inference.py` exits with code `0`.
2. The output contains at least 50 generated tokens.
3. Peak RSS does not exceed 14 GB (hard ceiling, system-stability guard).
4. No Python exception traceback appears in stderr.

---

## 6. Out of Scope

The following items are explicitly excluded from EX05:

- **Serving infrastructure:** No HTTP API, gRPC endpoint, or WebSocket streaming. This is a local CLI tool only.
- **Multi-GPU or distributed inference:** The target environment has at most one iGPU; sharding across devices is not addressed.
- **Fine-tuning or training:** Only inference (forward pass) is in scope. No gradient computation.
- **Model training data or licensing compliance auditing:** Responsibility for gated model access (e.g., Llama 2 EULA) rests with the individual user.
- **Windows-native (non-WSL) execution:** The pipeline is tested under WSL2/Ubuntu only.
- **Production hardening:** No authentication, rate limiting, logging infrastructure, or SLA guarantees.

---

## 7. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Model download fails (HF gating / network) | Medium | High | Use an openly licensed fallback model (e.g., `microsoft/phi-2`); document `.env` setup clearly |
| `bitsandbytes` CPU build unavailable for target Python version | Medium | High | Pin `bitsandbytes` to a known-working release in `uv.lock`; document fallback to fp16 if INT4 fails |
| Disk I/O bottleneck causes per-token latency > 120 s | Low | Medium | Document expected latency range; recommend NVMe over HDD; latency is a characterization target, not a hard SLA |
| WSL2 RAM limit (`wsl2.memory` in `.wslconfig`) set below 12 GB by user | Medium | High | Add a pre-flight check in `run_inference.py` that reads `/proc/meminfo` and warns if available RAM < 10 GB |
| `airllm` API breaks on a new `transformers` release | Low | High | Pin `transformers` and `airllm` in `uv.lock`; monitor `airllm` changelog |

---

*End of Document — ZeroVRAM-AirLLM-Engine PRD v1.0.0*
