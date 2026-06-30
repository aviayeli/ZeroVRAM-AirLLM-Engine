# Project TODO / Task Board
## ZeroVRAM-AirLLM-Engine (EX05)

**Version:** 1.0.0
**Status:** In Progress
**Author:** Avi Ayeli
**Date:** 2026-06-30
**Traceability:** Derived from [`docs/PRD.md`](./PRD.md) v1.0.0 and [`docs/PLAN.md`](./PLAN.md) v1.1.0
**Standard:** Dr. Segal's V3 Software Submission Guidelines

---

## How to Read This Board

- `[x]` = Done and verified against its Definition of Done (DoD).
- `[ ]` = Not started or in progress.
- Each **Milestone** carries an explicit **Definition of Done** — a milestone is not checked off until every task under it is checked off *and* the DoD condition is independently verifiable (a file exists, a command exits `0`, a number is recorded).
- Phase order is sequential and gated, per `docs/PLAN.md` §1 — do not begin a phase whose entry criteria depend on an unchecked milestone above it.

### Progress Summary

| Phase | Status | Completion |
|-------|--------|------------|
| 1 — Planning & Architecture | ✅ Complete | 100% (3/3 milestones) |
| 2 — Environment & Infrastructure | 🔶 In Progress | ~70% (core scaffolding, deps, secrets, config done; pre-flight check pending) |
| 3 — Baseline Execution | ⬜ Not Started | 0% |
| 4 — AirLLM Execution | ⬜ Not Started | 0% |
| 5 — Metrics & Visualization | ⬜ Not Started | 0% |
| 6 — FinOps & Economic Analysis | ⬜ Not Started | 0% |
| 7 — Final Reporting | ⬜ Not Started | 0% |

---

## Phase 1: Planning & Architecture

**Goal:** Establish the requirements and execution plan before any code is written.

### Milestone 1.1 — Product Requirements Document
- [x] Define project overview and EX05 classification (`docs/PRD.md` §1)
- [x] Document the zero-VRAM bottleneck and problem statement (§2)
- [x] Define goals, objectives (O1–O6), and baseline-vs-optimized comparison table (§3)
- [x] Specify system requirements: host environment, software dependencies, project tooling, target directory structure (§4)
- [x] Define success metrics: stability, latency, reproducibility, acceptance criteria (§5)
- [x] Define out-of-scope boundaries (§6)
- [x] Document risks and mitigations (§7)

**Definition of Done:** `docs/PRD.md` exists, is versioned (v1.0.0), and contains all seven sections above with no placeholder text. ✅ **Verified.**

### Milestone 1.2 — Implementation Plan
- [x] Decompose PRD into six gated phases with entry/exit criteria (`docs/PLAN.md` §1)
- [x] Detail Phase 1 environment setup commands (§2)
- [x] Detail Phase 2 model selection, quantization, and weight-format (SafeTensors vs. GGUF) decisions (§3)
- [x] Detail Phase 3 baseline OOM-demonstration steps (§4)
- [x] Detail Phase 4 AirLLM execution steps and fixed prompt set (§5)
- [x] Detail Phase 5 metrics, evaluation, and reporting steps (§6)
- [x] Detail Phase 6 economic analysis steps and comparison scenarios (§7)
- [x] Define project-level Definition of Done and risk carry-forward table (§8–9)

**Definition of Done:** `docs/PLAN.md` exists, is versioned (v1.1.0), traces every section back to a PRD objective, and covers all six execution phases. ✅ **Verified.**

### Milestone 1.3 — Task Board
- [x] Generate `docs/TODO.md` (this document) from PRD + PLAN
- [x] Review task board against PRD §5.4 acceptance criteria for full coverage

**Definition of Done:** `docs/TODO.md` committed, every PLAN.md phase has a corresponding Phase section below with granular, checkbox-tracked tasks. ✅ **Verified.**

---

## Phase 2: Environment & Infrastructure

**Goal:** A reproducible, `uv`-managed environment that a new contributor can stand up in ≤ 10 minutes (PRD §5.3).

> **Architecture note:** Per explicit direction for this phase, shared application configuration lives at the top-level `src/config.py` (using `pydantic_settings.BaseSettings`), rather than nested under `src/engine/`. Engine-specific modules (`loader.py`, `infer.py`, `benchmark.py`) will be created under `src/engine/` in Phases 3–5 and will import shared settings from `src.config`. This supersedes the flat-constants `config.py` sketch originally outlined in `docs/PLAN.md` §2.

### Milestone 2.1 — Project Scaffolding
- [x] Run `uv python pin 3.12` → `.python-version` created
- [x] Run `uv init --name zerovram-airllm-engine --no-readme --no-workspace` (the user's home directory has an unrelated parent `uv` workspace; `--no-workspace` was required to keep this project standalone)
- [x] Run `uv venv` → `.venv/` created (via first `uv add`) and confirmed gitignored
- [ ] Create `src/engine/{__init__.py,loader.py,infer.py,benchmark.py}` and `scripts/run_inference.py` — **deferred to Phase 3/4**, created when the baseline and AirLLM scripts are actually implemented, not as empty stubs ahead of need

**Definition of Done:** `uv venv` exits `0`; `.venv/` is gitignored. ✅ **Verified** (engine module stubs intentionally deferred — see note above).

### Milestone 2.2 — Dependency Installation
- [x] `uv add torch --index-url https://download.pytorch.org/whl/cpu` (persisted as an explicit `[[tool.uv.index]]` + `[tool.uv.sources]` entry in `pyproject.toml` so a plain `uv sync` reproduces the CPU build with no manual flag)
- [x] `uv add airllm transformers accelerate pydantic-settings python-dotenv`
- [ ] `uv add bitsandbytes` — deferred to Phase 4 (only needed once 4-bit quantized AirLLM execution is implemented)
- [ ] `uv add --dev psutil pytest ruff` — deferred: `psutil` lands with the Phase 3 RSS sampler, `pytest`/`ruff` with the first testable module
- [ ] `uv add --dev matplotlib seaborn pandas` — deferred to Phase 5 (Metrics & Visualization)
- [x] `uv lock` → `uv.lock` committed
- [x] `uv sync` → exits `0` on a clean checkout

**Definition of Done:** `uv sync` exits `0`; `uv run python -c "import torch, airllm, transformers, accelerate; print('ok')"` prints `ok` with no traceback; `uv.lock` is committed and matches `pyproject.toml`. ✅ **Verified for currently-installed deps.** (`bitsandbytes`, dev tooling, and visualization deps will be re-verified as a single combined import check once Phases 3–5 add them.)

### Milestone 2.3 — Hugging Face Authentication
- [x] Create `.env-example` with `HUGGING_FACE_TOKEN=` placeholder
- [x] Confirm `.env` is excluded via `.gitignore` (verified with `git check-ignore -v .env`)
- [ ] Populate local `.env` with a real, valid HF token — **user action required**; the agent cannot generate or possess this secret
- [ ] Verify auth: `uv run python -c "from huggingface_hub import whoami; print(whoami())"` — blocked on the above

**Definition of Done:** `.env-example` exists with no real secret; `.env` confirmed absent from `git status` / `git ls-files`. ⏳ **`whoami()` verification pending user-supplied token.**

### Milestone 2.4 — `config.py` (Centralized Configuration)
- [x] Create `src/config.py` using `pydantic_settings.BaseSettings`
- [x] Securely load `hugging_face_token` as `pydantic.SecretStr` (required field, sourced from `.env`/environment — never hardcoded, never logged in plaintext)
- [x] Define base paths as `pathlib.Path` fields, all derived dynamically from `Path(__file__).resolve().parent.parent` (no hardcoded absolute paths): `project_root`, `models_dir`, `logs_dir`, `results_dir`, `hf_cache_dir`
- [x] `ensure_directories()` helper to create the runtime data dirs on demand
- [x] `get_settings()` — `lru_cache`-wrapped singleton accessor
- [ ] `DEFAULT_MODEL_ID` / `FALLBACK_MODEL_ID` / `QUANT_CONFIG` / `MIN_AVAILABLE_RAM_GB` — deferred to Phase 3/4 (Model Selection / AirLLM Execution), where they are actually consumed

**Definition of Done:** `HUGGING_FACE_TOKEN=<value> uv run python -c "from src.config import get_settings; s = get_settings(); print(s.project_root)"` runs with no error and resolves to the repo root with no hardcoded path; file is 41 lines (well under the 150-line V3 ceiling). ✅ **Verified.**

### Milestone 2.5 — RAM Pre-Flight Check
- [ ] Implement a `check_available_ram()` utility reading `/proc/meminfo`
- [ ] Wire warning (not hard block) if `MemAvailable` < `MIN_AVAILABLE_RAM_GB` (PRD §7 mitigation)

**Definition of Done:** Function returns the actual available RAM in GB and logs a warning when below threshold; unit-tested with a mocked `/proc/meminfo` fixture. **Not started — carried to Phase 3.**

---

## Phase 3: Baseline Execution

**Goal:** Empirically demonstrate the memory-bound failure mode described in PRD §2.1–§2.2.

### Milestone 3.1 — Native Loading Script
- [ ] Implement `scripts/run_baseline.py`
- [ ] Load selected model via plain `transformers.AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.float16)` — **no AirLLM, no quantization**
- [ ] Wire a background RSS sampler (1 Hz, via `psutil`) writing to `results/baseline_rss.csv`
- [ ] Wire stderr/stdout capture to `results/baseline_stderr.log`

**Definition of Done:** `scripts/run_baseline.py` runs end-to-end (regardless of crash/success) and always leaves behind a partial-or-complete `results/baseline_rss.csv`, even on `SIGKILL`.

### Milestone 3.2 — Execute and Capture Failure
- [ ] Run: `uv run python scripts/run_baseline.py 2>&1 | tee results/baseline_stderr.log`
- [ ] Record exit code (`echo $?`)
- [ ] Confirm failure signature: `MemoryError`, allocator `RuntimeError`, or OOM-killer `SIGKILL` (exit 137)
- [ ] **If the load unexpectedly succeeds:** escalate to `torch_dtype=torch.float32` (28 GB footprint) to force the bottleneck deterministically; document the escalation

### Milestone 3.3 — Crawl-Path Contingency (If No Crash Occurs)
- [ ] If the baseline does **not** crash, measure TTFT (`time.perf_counter()` delta to first token) and TPOT on the slow CPU-resident path
- [ ] Record these "best case, no paging" numbers in `results/baseline_metrics.csv` for use as an upper-bound reference in Phase 5
- [ ] Explicitly document in `docs/REPORT.md` why the host did not exhibit the expected OOM (e.g., more RAM available than the 16 GB target spec)

**Definition of Done (Phase 3):** Exactly one of the following is true and is backed by a committed artifact: (a) `results/baseline_stderr.log` shows a captured crash with non-zero/137 exit code, **or** (b) `results/baseline_metrics.csv` records TTFT/TPOT for a successful-but-degraded run, with the deviation explained in the report. No silent, undocumented outcome is acceptable.

---

## Phase 4: AirLLM Execution

**Goal:** Crash-free, layer-paged inference on the target hardware (PRD primary goal, §3.1).

### Milestone 4.1 — Model Loader
- [ ] Implement `src/engine/loader.py`
- [ ] Wrap `airllm.AutoModel.from_pretrained(model_id, compression="4bit")`
- [ ] Expose single typed `load_model() -> AirLLMModel` function (no bare `Any` return type)
- [ ] Use `DEFAULT_MODEL_ID` / `QUANT_CONFIG` from `src/engine/config.py` — no hardcoded model strings

**Definition of Done:** `uv run python -c "from src.engine.loader import load_model; m = load_model(); print(type(m))"` succeeds without raising.

### Milestone 4.2 — Inference Loop
- [ ] Implement `src/engine/infer.py`
- [ ] Tokenize the fixed prompt set (P1–P3 from `docs/PLAN.md` §5.3)
- [ ] Drive `model.generate(...)` through AirLLM's layer-by-layer mmap path
- [ ] Yield/stream decoded tokens (not just final string) so TTFT can be measured at the first token boundary

### Milestone 4.3 — CLI Entrypoint
- [ ] Implement `scripts/run_inference.py`
- [ ] Args: `--prompt`, `--max-new-tokens`, `--model-id` (default from config)
- [ ] Wire in the RAM pre-flight check from Milestone 2.5
- [ ] Wire in the RSS sampler (reused from Phase 3) → `results/airllm_rss.csv`
- [ ] Wire in a VRAM sampler (`intel_gpu_top -J` or `/sys/class/drm/card*/device/mem_info_vram_used`) → `results/airllm_vram.csv`

**Definition of Done:** `uv run python scripts/run_inference.py --prompt "Explain the concept of zero-VRAM inference." --max-new-tokens 50` exits `0`, prints ≥ 50 generated tokens, and produces both `results/airllm_rss.csv` and `results/airllm_vram.csv`.

### Milestone 4.4 — Stability Validation
- [ ] Run the CLI entrypoint **5 consecutive times** with the same prompt (PRD §5.1 acceptance test)
- [ ] Log each run's timestamp, exit code, and output token count to `results/stability_log.txt`
- [ ] Confirm 0 OOM crashes across all 5 runs
- [ ] Confirm peak RSS ≤ 14 GB (hard ceiling) on every run

**Definition of Done (Phase 4):** `results/stability_log.txt` shows 5/5 runs with exit code `0`, ≥ 50 tokens each, no stderr traceback, peak RSS ≤ 14 GB, and VRAM usage explicitly recorded (even if `0 MB`) for every run.

---

## Phase 5: Metrics & Visualization

**Goal:** Turn raw run logs into a quantified, visual Baseline-vs-AirLLM comparison.

### Milestone 5.1 — Benchmark Utilities
- [ ] Implement `src/engine/benchmark.py`
- [ ] `measure_ttft()` — `time.perf_counter()` delta from `generate()` call to first decoded token
- [ ] `measure_tpot()` — `(output_tokens - 1) / (t_end - t_first_token)`
- [ ] `sample_rss()` — peak `VmRSS` from `/proc/self/status`, 1 Hz
- [ ] `sample_vram()` — peak iGPU memory allocation, 1 Hz
- [ ] Refactor `scripts/run_baseline.py` and `scripts/run_inference.py` to import from this module (remove ad-hoc duplication)

**Definition of Done:** `src/engine/benchmark.py` is the single source of truth for all measurement logic; no `time.perf_counter()` or `/proc` reads exist outside this module.

### Milestone 5.2 — Data Collection
- [ ] Run TTFT/TPOT trials for prompts P1–P3, 3 trials each, on the AirLLM path
- [ ] Aggregate into `results/metrics.csv` with columns: `prompt_id, trial, ttft_s, tpot_tok_s, peak_rss_gb, peak_vram_mb, exit_code`
- [ ] Merge in `results/baseline_metrics.csv` (if Milestone 3.3 applies) for direct comparison rows

**Definition of Done:** `results/metrics.csv` has no empty cells, no "TBD" values, and covers every prompt × trial combination defined in the plan.

### Milestone 5.3 — Comparison Table
- [ ] Reproduce PRD §3.3's Baseline-vs-AirLLM table with **actual measured numbers**
- [ ] Include: peak RAM, VRAM, OOM crash rate, TTFT, sustained tokens/sec

### Milestone 5.4 — Visualization (Matplotlib / Seaborn)
- [ ] Create `scripts/generate_charts.py` (or `notebooks/analysis.ipynb`)
- [ ] Chart 1: Bar chart — Peak RAM usage, Baseline vs. AirLLM
- [ ] Chart 2: Bar chart — Peak VRAM usage, Baseline vs. AirLLM
- [ ] Chart 3: Line/bar chart — TTFT per prompt (P1–P3), AirLLM only (or both, if baseline data exists)
- [ ] Chart 4: Line chart — RSS growth over time (sampled curve) for Baseline vs. AirLLM, overlaid
- [ ] Save all charts to `docs/assets/` as `.png`, referenced from `docs/REPORT.md`

**Definition of Done (Phase 5):** `results/metrics.csv` is fully populated; `docs/assets/` contains all 4 charts as committed `.png` files; every chart is referenced by filename in the final report (Phase 7).

---

## Phase 6: FinOps & Economic Analysis

**Goal:** Quantify the cost trade-off of on-premises zero-VRAM inference vs. cloud/API alternatives (`docs/PLAN.md` §7).

### Milestone 6.1 — Scenario Definition
- [ ] Scenario A — On-Premises: amortized hardware (sunk cost) + electricity only
- [ ] Scenario B — Hosted Inference API: published $/M input + $/M output token pricing for a comparable 7–8B-class model
- [ ] Scenario C — Cloud GPU Rental: published on-demand $/hour rate for a mid-tier datacenter GPU, converted to $/1K tokens

### Milestone 6.2 — Cost Calculations
- [ ] Compute Scenario A electricity cost: `(host TDP in kW) × (wall-clock seconds per run / 3600) × (local $/kWh)` — cite the $/kWh source
- [ ] Record Scenario B pricing with source URL + retrieval date
- [ ] Record Scenario C pricing with source URL + retrieval date; derive $/1K tokens from documented/benchmarked GPU throughput
- [ ] Normalize all three scenarios to **$ per 1,000 output tokens** using Phase 5's measured TPOT for Scenario A

### Milestone 6.3 — Break-Even Analysis
- [ ] Build the comparison table: `Scenario | $/1K tokens | Latency class (TTFT) | Setup overhead | Notes`
- [ ] Determine the monthly token volume at which Scenario A's marginal cost beats Scenarios B/C
- [ ] Frame explicitly as a **latency-for-cost trade-off**, not a blanket "cheaper" claim (per `docs/PLAN.md` §7.5)

**Definition of Done (Phase 6):** All three scenarios quantified in the same unit with every external number cited and date-stamped ("as of 2026-06-30" or actual retrieval date); break-even volume stated as a concrete number, not a vague qualitative claim.

---

## Phase 7: Final Reporting

**Goal:** A deep-dive technical report that explains both the *what* (results) and the *why* (architecture) of the system.

### Milestone 7.1 — Architecture & Concepts Explainer (README.md)
- [ ] Write/update `README.md` project overview section
- [ ] Explain **disk paging** as applied to model weights (why layer-by-layer loading avoids full-model RAM residency)
- [ ] Explain **mmap** (memory-mapped I/O) and why SafeTensors' zero-copy mmap interface is what makes AirLLM's layer-paging viable (per `docs/PLAN.md` §3.4)
- [ ] Explain the 4-bit quantization strategy and its effect on per-layer footprint
- [ ] Include an architecture diagram (ASCII or image) showing: disk → mmap → single-layer-in-RAM → forward pass → discard → next layer

### Milestone 7.2 — Deep-Dive Technical Report (`docs/REPORT.md`)
- [ ] Section: Methodology
- [ ] Section: Hardware/Software Configuration (exact specs used for the actual run, not just the target spec)
- [ ] Section: Baseline Failure Analysis (embed `results/baseline_stderr.log` excerpt or the crawl-path metrics from Milestone 3.3)
- [ ] Section: AirLLM Results (stability log summary, 5/5 run table)
- [ ] Section: TTFT / TPOT / RAM / VRAM Tables + analysis (embed Phase 5 charts)
- [ ] Section: Economic Analysis (embed Phase 6 comparison table and break-even statement)
- [ ] Section: Limitations
- [ ] Section: Conclusion

### Milestone 7.3 — Acceptance Cross-Check
- [ ] Walk PRD §5.4 acceptance criteria line by line; mark each pass/fail with an evidence pointer (file + line, or results row)
- [ ] Walk `docs/PLAN.md` §8 Definition of Done line by line; confirm every item checked
- [ ] Confirm `uv sync` + `uv run python scripts/run_inference.py` still succeeds on a clean clone within 10 minutes

**Definition of Done (Phase 7):** `README.md` and `docs/REPORT.md` both exist, contain no "TODO"/"TBD" placeholders, every referenced chart/log file is present in the repo, and the acceptance cross-check table shows 100% pass with evidence pointers.

---

## Final Project Definition of Done

The EX05 submission is complete when **every** box above is checked **and**:

- [ ] `uv sync` exits `0` on a fresh clone
- [ ] `scripts/run_inference.py` runs crash-free, 5/5, with logged evidence
- [ ] `results/metrics.csv` and all four Phase 5 charts are committed
- [ ] `docs/REPORT.md` includes Economic Analysis with cited, dated sources
- [ ] `README.md` explains paging and mmap clearly enough for a reader unfamiliar with AirLLM to understand the architecture
- [ ] No `pip`/`conda` invocation appears anywhere in the repo
- [ ] All P0/P1 PRD objectives (O1–O5) are evidence-linked; O6 is evidence-linked via the final report

---

*End of Document — ZeroVRAM-AirLLM-Engine Project TODO Board v1.0.0*
