# Implementation Plan
## ZeroVRAM-AirLLM-Engine (EX05)

**Version:** 1.1.0
**Status:** Draft
**Author:** Avi Ayeli
**Date:** 2026-06-30
**Traceability:** Implements [`docs/PRD.md`](./PRD.md) v1.0.0

---

## Table of Contents

1. [Plan Overview](#1-plan-overview)
2. [Phase 1 — Environment Setup](#2-phase-1--environment-setup)
3. [Phase 2 — Model Selection](#3-phase-2--model-selection)
4. [Phase 3 — Baseline Execution (OOM Demonstration)](#4-phase-3--baseline-execution-oom-demonstration)
5. [Phase 4 — AirLLM Execution](#5-phase-4--airllm-execution)
6. [Phase 5 — Metrics, Evaluation & Reporting](#6-phase-5--metrics-evaluation--reporting)
7. [Phase 6 — Economic Analysis](#7-phase-6--economic-analysis)
8. [Definition of Done](#8-definition-of-done)
9. [Risk Carry-Forward](#9-risk-carry-forward)

---

## 1. Plan Overview

This plan decomposes the PRD into six sequential, gated phases. Each phase has explicit **entry criteria**, **exit criteria**, **deliverables**, and **commands**. A phase may not start until the prior phase's exit criteria are met — this is a linear experiment pipeline, not a parallelizable workstream, because Phase 3's failure mode (the OOM crash) is the empirical baseline that Phase 4 is measured against, and Phase 6's cost model depends on Phase 5's measured throughput.

| Phase | Name | Maps to PRD Objective(s) | Est. Effort |
|-------|------|---------------------------|--------------|
| 1 | Environment Setup | O5 | 0.5 day |
| 2 | Model Selection | O1, O2 | 0.5 day |
| 3 | Baseline Execution (OOM Demonstration) | O4 | 0.5 day |
| 4 | AirLLM Execution | O1, O2, O3 | 1 day |
| 5 | Metrics, Evaluation & Reporting | O3, O4, O6 | 1 day |
| 6 | Economic Analysis | O6 (extension) | 0.5 day |

**Engineering standard:** All commands in this plan use `uv` exclusively (per PRD §4.3 — `pip`/`conda` are disallowed). All scripts are typed (PEP 604 unions, no bare `Any` in public signatures), fail loudly (no silent `except: pass`), and every phase produces a durable artifact under `docs/` or `results/` so the experiment is auditable after the fact.

---

## 2. Phase 1 — Environment Setup

### 2.1 Entry Criteria
- Fresh WSL2/Ubuntu 22.04 instance per PRD §4.1.
- Repository cloned; current directory is repo root.

### 2.2 Steps

| # | Step | Command |
|---|------|---------|
| 1.1 | Verify `uv` is installed | `uv --version` (install via `curl -LsSf https://astral.sh/uv/install.sh \| sh` if absent) |
| 1.2 | Pin Python version | `uv python pin 3.12` → writes `.python-version` |
| 1.3 | Initialize project metadata (if `pyproject.toml` absent) | `uv init --name zerovram-airllm-engine --no-readme` |
| 1.4 | Create the virtual environment | `uv venv` |
| 1.5 | Add core runtime dependencies | `uv add torch --index-url https://download.pytorch.org/whl/cpu` |
| 1.6 | Add inference stack | `uv add airllm transformers accelerate bitsandbytes huggingface-hub` |
| 1.7 | Add dev/benchmark tooling | `uv add --dev psutil pytest ruff` |
| 1.8 | Generate and commit lock file | `uv lock` → commit `uv.lock` |
| 1.9 | Scaffold target directory structure (PRD §4.4) | create `src/engine/{__init__.py,loader.py,infer.py,benchmark.py}`, `scripts/run_inference.py`, `.env.example` |
| 1.10 | Configure HF auth template | `.env.example` containing `HF_TOKEN=` (gitignored `.env` holds the real value) |
| 1.11 | Add `.wslconfig` RAM pre-flight note to README | document the 12 GB `wsl2.memory` requirement from PRD §7 |
| 1.12 | Sync and smoke-test | `uv sync` then `uv run python -c "import torch, airllm, transformers; print('ok')"` |

### 2.3 Deliverables
- `pyproject.toml`, `uv.lock` committed.
- `src/engine/` and `scripts/` scaffolded with empty-but-importable modules.
- `.env.example` present; `.env` gitignored.

### 2.4 Exit Criteria (Gate to Phase 2)
- `uv sync` exits `0` on a clean checkout.
- `uv run python -c "import torch, airllm, transformers; print('ok')"` prints `ok` with no traceback.
- Setup time from `git clone` to this point ≤ 10 minutes (PRD §5.3), recorded in `docs/PLAN.md` execution log (Appendix, §7).

---

## 3. Phase 2 — Model Selection

### 3.1 Candidate Models

| Candidate | Params | Native fp16 Size | 4-bit (INT4) Size | License | Selection Notes |
|-----------|--------|-------------------|---------------------|---------|------------------|
| `meta-llama/Meta-Llama-3-8B-Instruct` | 8B | ~16 GB | ~4.5–5 GB | Llama 3 Community License (gated, requires HF approval) | Primary candidate; strong instruction-following baseline |
| `Qwen/Qwen2.5-7B-Instruct` | 7.6B | ~15 GB | ~4.3 GB | Apache 2.0 (ungated) | **Fallback / default** — no gating delay, unblocks Phase 2 immediately |
| `microsoft/phi-2` | 2.7B | ~5.4 GB | ~1.6 GB | MIT (ungated) | Emergency fallback only if both 7–8B candidates are infeasible within the 16 GB RAM ceiling (PRD §7 risk) |

### 3.2 Decision

Default to **`Qwen2.5-7B-Instruct`** for the primary experiment run (ungated → zero setup friction, satisfies PRD §5.3 reproducibility target). Attempt `Meta-Llama-3-8B-Instruct` as a secondary run **only if** `HF_TOKEN` gating is approved before Phase 4 begins, since it is explicitly named in the PRD's experiment brief. Both are 7–8B class and quantize to a 4-bit footprint comfortably under the 8 GB RAM target in PRD §3.3.

### 3.3 Quantization Strategy

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Precision | INT4 (NF4 via `bitsandbytes`) | Matches PRD O2; reduces per-layer footprint ~4x vs. fp16 |
| Compute dtype | `bfloat16` (CPU-compatible accumulation) | Avoids fp16 underflow on CPU paths |
| Quantization scope | All transformer block linear layers; embeddings/lm_head left at higher precision | AirLLM default behavior; preserves output quality at the vocabulary projection |
| Compression flag | `compression="4bit"` passed to `AutoModel.from_pretrained` (AirLLM API) | Required for AirLLM's layer-paging quantized path used in Phase 4 |

### 3.4 Weight Format Selection (SafeTensors vs. GGUF)

| Format | Use With | Decision |
|--------|----------|----------|
| **SafeTensors** | AirLLM's native `transformers`-based layer-splitting path (`AutoModel.from_pretrained`) | **Primary format.** AirLLM's mmap-based layer-by-layer loader (Phase 4) is built on the SafeTensors zero-copy mmap interface — this is the format that makes the on-demand paging strategy work at all, since each shard can be mapped independently without deserializing the full checkpoint |
| **GGUF** | `llama.cpp`-style CPU-quantized inference (e.g., `llama-cpp-python`) | **Evaluated as a secondary/comparison path only**, not the primary pipeline. GGUF's block-quantized (Q4_K_M) format is a legitimate alternative for fitting an 8B model in 16 GB RAM, and may be benchmarked in Phase 5 as a reference point against AirLLM, but it bypasses AirLLM's layer-paging mechanism entirely and is therefore out of scope for Phase 4's core experiment |

**Rationale:** SafeTensors is required for the AirLLM mmap path that this project exists to validate (PRD §1, §2.3). GGUF is documented here so the choice is explicit and auditable, not because both formats are used interchangeably — using GGUF would mean testing `llama.cpp`, not AirLLM.

### 3.5 Steps

| # | Step |
|---|------|
| 2.1 | Set `HF_TOKEN` in `.env`; run pre-flight `huggingface-cli whoami` equivalent via `uv run python -c "from huggingface_hub import whoami; print(whoami())"` |
| 2.2 | Download/cache tokenizer + config only (no weights yet) for both candidates to confirm repo access |
| 2.3 | Confirm each candidate repo publishes SafeTensors weights (not pickle-only `.bin`) — required for the AirLLM mmap path in Phase 4 |
| 2.4 | Record disk footprint estimate per candidate against the ≥ 30 GB free space requirement (PRD §4.1) |
| 2.5 | Document final model choice, format decision, and fallback chain in `docs/PLAN.md` execution log |

### 3.6 Deliverables
- `src/engine/loader.py`: model name + quantization config defined as named constants (e.g., `DEFAULT_MODEL_ID`, `FALLBACK_MODEL_ID`, `QUANT_CONFIG`).

### 3.7 Exit Criteria (Gate to Phase 3)
- Selected model ID(s) confirmed accessible (tokenizer/config fetch succeeds, no 401/403) and confirmed to publish SafeTensors weights.
- Disk space verified sufficient for full fp16 download (needed for Phase 3's native attempt) **and** the quantized cache.

---

## 4. Phase 3 — Baseline Execution (OOM Demonstration)

This phase exists to **empirically demonstrate the failure mode** described in PRD §2.1–§2.2 — it is intentionally expected to fail. The failure artifact (full traceback, RSS curve up to the kill point) is itself a required deliverable for the final report (Phase 5).

### 4.1 Entry Criteria
- Phase 2 complete; model ID finalized.
- `psutil`-based memory sampler available (added in Phase 1.7).

### 4.2 Steps

| # | Step | Detail |
|---|------|--------|
| 3.1 | Implement `scripts/run_baseline.py` | Loads the selected model via plain `transformers.AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.float16)` — **no** AirLLM, **no** quantization, mirroring PRD §2.2's naive path |
| 3.2 | Wrap the load call with a background RSS sampler (1 Hz, per PRD §5.2) writing to `results/baseline_rss.csv` until process exit or kill |
| 3.3 | Execute under a bounded environment | Run inside the documented 16 GB WSL2 guest so the OOM condition is reachable without harming the Windows host: `uv run python scripts/run_baseline.py 2>&1 \| tee results/baseline_stderr.log` |
| 3.4 | Capture the failure signature | Expect one of: `MemoryError`, `RuntimeError` (allocator failure), or OS OOM-killer SIGKILL (exit code 137) |
| 3.5 | If the host happens to have ≥ 14 GB free and the load *succeeds*, escalate the test: run a second baseline attempt with `torch_dtype=torch.float32` (28 GB footprint) to force the bottleneck deterministically, and note the dtype escalation in the report |

### 4.3 Deliverables
- `scripts/run_baseline.py`
- `results/baseline_rss.csv` — RAM growth curve up to crash point
- `results/baseline_stderr.log` — full traceback / kill signal evidence
- Recorded exit code (expect non-zero / 137)

### 4.4 Exit Criteria (Gate to Phase 4)
- The OOM (or equivalent resource-exhaustion) failure is reproduced and captured with logs, **or** explicitly documented as non-reproducible on the available test host with the actual peak RSS reached — either outcome is an acceptable, evidence-backed exit, per PRD §3.3's "N/A (crashes before inference)" expectation.
- Baseline artifacts committed to `results/` (or `results/` added to `.gitignore` with artifacts referenced from the report instead, if file size is prohibitive — decide in Phase 5).

---

## 5. Phase 4 — AirLLM Execution

### 5.1 Entry Criteria
- Phase 3 complete; baseline failure (or peak-RSS ceiling) documented.
- Quantization config finalized in Phase 2.

### 5.2 Steps

| # | Step | Detail |
|---|------|--------|
| 4.1 | Implement `src/engine/loader.py` | Wraps `airllm.AutoModel.from_pretrained(model_id, compression="4bit")`; exposes a single `load_model()` function with explicit return type |
| 4.2 | Implement `src/engine/infer.py` | Tokenizes a fixed prompt set (see §5.3), drives `model.generate(...)` via AirLLM's layer-by-layer mmap path, yields decoded tokens |
| 4.3 | Implement `scripts/run_inference.py` | CLI entrypoint per PRD §4.4; arguments: `--prompt`, `--max-new-tokens`, `--model-id` (defaults to Phase 2 selection) |
| 4.4 | Add the RAM pre-flight check | Read `/proc/meminfo`; warn (not block) if `MemAvailable` < 10 GB, per PRD §7 mitigation |
| 4.5 | Wire in the RSS sampler from Phase 3 | Reused for AirLLM run so baseline vs. optimized curves are directly comparable |
| 4.6 | First crash-free run | `uv run python scripts/run_inference.py --prompt "Explain the concept of zero-VRAM inference." --max-new-tokens 50` |
| 4.7 | Stability validation | Repeat step 4.6 five consecutive times per PRD §5.1's acceptance test; record exit code each run |

### 5.3 Fixed Prompt Set (used identically in Phase 5 benchmarking)

| ID | Prompt | Purpose |
|----|--------|---------|
| P1 | "Explain the concept of zero-VRAM inference." | Short-context, open-ended generation |
| P2 | "Summarize the following in two sentences: [200-word passage]" | Longer-context comprehension |
| P3 | "Write a Python function to reverse a linked list." | Code-generation latency profile |

### 5.4 Deliverables
- `src/engine/loader.py`, `src/engine/infer.py`, `scripts/run_inference.py`
- `results/airllm_rss.csv` per run
- 5/5 successful run log (`results/stability_log.txt`: timestamp, exit code, output token count)

### 5.5 Exit Criteria (Gate to Phase 5)
- 0 OOM crashes across 5 consecutive runs (PRD §5.1 target).
- Each run exits code `0`, produces ≥ 50 generated tokens, no stderr traceback (PRD §5.4 acceptance criteria 1–4).
- Peak RSS ≤ 14 GB hard ceiling on every run.

---

## 6. Phase 5 — Metrics, Evaluation & Reporting

### 6.1 Entry Criteria
- Phase 4 complete; 5/5 stable runs achieved.

### 6.2 Steps

| # | Step | Detail |
|---|------|--------|
| 5.1 | Implement `src/engine/benchmark.py` | Defines `measure_ttft()`, `measure_tpot()`, `sample_rss()`, and `sample_vram()` as composable utilities (no duplication with Phase 3/4 ad-hoc sampling — Phase 3/4 scripts should import from this module once it exists) |
| 5.2 | TTFT measurement | `time.perf_counter()` delta from `generate()` call to first decoded token, per prompt P1–P3, averaged over 3 trials each |
| 5.3 | TPOT measurement | `(output_tokens - 1) / (t_end - t_first_token)`, same trial structure |
| 5.4 | RAM measurement | Peak `VmRSS` from `/proc/self/status`, sampled at 1 Hz throughout each trial |
| 5.5 | VRAM measurement | Sample Intel iGPU memory allocation via `intel_gpu_top -J` (or `/sys/class/drm/card*/device/mem_info_vram_used` if exposed) at 1 Hz throughout each trial, confirming usage stays within the ≤ 128 MB ceiling per PRD §4.1; record `0 MB` explicitly if AirLLM never offloads to the iGPU on this run, since that is itself a meaningful result for the zero-VRAM thesis |
| 5.6 | Aggregate results | Write `results/metrics.csv` with columns: `prompt_id, trial, ttft_s, tpot_tok_s, peak_rss_gb, peak_vram_mb, exit_code` |
| 5.7 | Baseline-vs-AirLLM comparison table | Reproduce PRD §3.3's table with **actual measured numbers** (including VRAM) replacing the "expected" placeholders |
| 5.8 | Write the deep-dive technical report | `docs/REPORT.md` — sections: Methodology, Hardware/Software Config, Baseline Failure Analysis (with `results/baseline_stderr.log` excerpt), AirLLM Results, TTFT/TPOT/RAM/VRAM Tables + analysis, Limitations, Conclusion |
| 5.9 | Cross-check against PRD Acceptance Criteria | Walk PRD §5.4 line by line; mark each criterion pass/fail with evidence pointer |

### 6.3 Deliverables
- `src/engine/benchmark.py`
- `results/metrics.csv`
- `docs/REPORT.md` (final deep-dive technical report — primary EX05 academic deliverable per PRD O6)

### 6.4 Exit Criteria (Gate to Phase 6)
- `docs/REPORT.md` published with all measured metrics, baseline failure evidence, and the comparison table fully populated (no "TBD" placeholders).
- All PRD §5.4 acceptance criteria evaluated with explicit pass/fail and evidence.
- `uv.lock` still in sync (`uv sync` exits `0`) — confirms PRD §5.3 reproducibility holds after the full experiment.
- Sustained tokens/sec from `results/metrics.csv` available as the throughput input to Phase 6's cost model.

---

## 7. Phase 6 — Economic Analysis

This phase is a desk-research extension, not a code deliverable: it converts Phase 5's measured throughput into a cost comparison so the report can answer "is this actually worth running locally?" alongside "does it technically work?" This directly serves PRD O6 (document the pipeline to a full academic deliverable standard) by adding the cost dimension the core PRD scopes out of stability/latency metrics but a deep-dive technical report should still address.

### 7.1 Entry Criteria
- Phase 5 complete; `results/metrics.csv` contains measured TPOT (tokens/sec) for the selected model.

### 7.2 Comparison Scenarios

| Scenario | Description | Cost Basis |
|----------|-------------|------------|
| **A — On-Premises (this project)** | Existing WSL2 laptop, AirLLM, 4-bit quantized 7–8B model | Amortized hardware (sunk cost, already owned) + electricity only |
| **B — Hosted Inference API** | Equivalent-class hosted model (e.g., a comparable 7–8B-class instruct model via a commercial inference API) | Published per-million-token pricing (input + output, priced separately) |
| **C — Cloud GPU Rental** | Same 7B-class model run full-precision/fp16 on a rented cloud GPU instance (e.g., a single mid-tier datacenter GPU, on-demand hourly) | Published on-demand $/hour rate from the cloud provider's public pricing page, converted to $/1K tokens using the instance's typical throughput for a 7–8B model |

### 7.3 Steps

| # | Step | Detail |
|---|------|--------|
| 6.1 | Compute on-prem electricity cost | `(host TDP in kW) × (wall-clock seconds per Phase 5 run / 3600) × (local $/kWh)`; use a documented, cited $/kWh figure rather than an assumed one |
| 6.2 | Collect Scenario B pricing | Record the provider's published $/M input tokens and $/M output tokens for a comparable model tier, with source URL and retrieval date, in `docs/REPORT.md` |
| 6.3 | Collect Scenario C pricing | Record the cloud provider's published on-demand hourly GPU rate, with source URL and retrieval date; derive $/1K tokens using that GPU's documented or benchmarked throughput for a 7–8B model |
| 6.4 | Normalize all three scenarios to a common unit | $ per 1,000 output tokens, using Phase 5's measured TPOT for Scenario A and the published/derived rates for B and C |
| 6.5 | Build the comparison table | Columns: `Scenario, $/1K tokens, Latency class (TTFT), Setup overhead, Notes` |
| 6.6 | State the break-even framing | At what monthly token volume does Scenario A's sunk-cost-only marginal cost become cheaper than B or C, given A's much lower throughput — present as a volume/latency trade-off, not a blanket "cheaper" claim |
| 6.7 | Append to the final report | Add a "§6 Economic Analysis" section to `docs/REPORT.md` (written in Phase 5.8) with the table, the break-even framing, and explicit pricing sources/dates so the numbers are falsifiable and re-checkable later |

### 7.4 Deliverables
- `docs/REPORT.md` §"Economic Analysis" section: normalized $/1K-token comparison table, electricity cost derivation, and cited source links/dates for Scenarios B and C.

### 7.5 Exit Criteria (Project Complete)
- All three scenarios (A, B, C) are quantified in the same unit ($/1K output tokens) with sources cited for every externally-sourced number.
- The trade-off is framed honestly: this project's on-prem path is a *latency-for-cost* trade against hosted alternatives, not a free win — the report must not claim cost superiority without the throughput caveat.
- No pricing figure in the report is older than the project's experiment window without an explicit "as of [date]" annotation.

---

## 8. Definition of Done

The project is considered complete when **all** of the following hold simultaneously:

- [ ] Phases 1–6 exit criteria each satisfied (see per-phase §2.4, §3.7, §4.4, §5.5, §6.4, §7.5).
- [ ] `docs/REPORT.md` exists, is internally consistent with `results/*.csv`, and includes the Economic Analysis section from Phase 6.
- [ ] `uv sync` + `uv run python scripts/run_inference.py` succeeds on a clean clone within the 10-minute PRD §5.3 budget.
- [ ] No `pip`/`conda` invocation appears anywhere in scripts, docs, or `pyproject.toml`.
- [ ] All P0/P1 objectives from PRD §3.2 (O1, O2, O3, O4, O5) are marked resolved with an evidence pointer into this plan or the report.

---

## 9. Risk Carry-Forward

Risks inherited from PRD §7 that materially affect execution of this plan; tracked here so phase owners check them at the relevant gate rather than re-discovering them mid-phase.

| Risk | Phase Affected | Trigger Check |
|------|-----------------|----------------|
| Gated model access delay (Llama 3) | Phase 2 | If `HF_TOKEN` approval is pending at the Phase 2→3 gate, proceed with `Qwen2.5-7B-Instruct` only and note the deferral |
| `bitsandbytes` CPU build unavailable | Phase 1, Phase 4 | Verify import succeeds in Phase 1.12; if INT4 path fails in Phase 4, fall back to fp16 AirLLM load and document the deviation in the report |
| Disk I/O latency > 120 s/token | Phase 4, Phase 5 | If observed, treat as a characterization data point, not a blocking failure — record and explain, do not retry indefinitely |
| WSL2 `.wslconfig` RAM cap < 12 GB | Phase 1, Phase 4 | Pre-flight check (step 4.4) surfaces this at runtime; if triggered, pause and have the user raise `wsl2.memory` before continuing |
| Published API/GPU pricing changes after report is written | Phase 6 | Date-stamp every external price cited; treat the cost table as a snapshot, not a live figure |

---

*End of Document — ZeroVRAM-AirLLM-Engine Implementation Plan v1.1.0*
