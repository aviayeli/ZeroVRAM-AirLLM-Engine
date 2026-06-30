"""Native (non-AirLLM) baseline: loads a full model into RAM via plain
`transformers` to empirically demonstrate the memory-bound bottleneck this
project exists to solve. Deliberately uses no AirLLM and no quantization.

Note: a true OOM-killer `SIGKILL` cannot be caught from inside Python — if
that happens, the process dies with exit code 137 and this script never gets
to log it. The `try/except` blocks below catch the *catchable* failure modes
(`MemoryError`, allocator `RuntimeError`, `KeyboardInterrupt`).
"""

import json
import logging
import resource
import sys
import threading
import time
import traceback
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from src.config import get_settings

PROMPT = "Explain the concept of Virtual Memory in Operating Systems."

logger = logging.getLogger("baseline")


def configure_logging(logs_dir: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(logs_dir / "baseline.log"), logging.StreamHandler(sys.stdout)],
    )


def load_model_and_tokenizer(model_id: str, hf_token: str, cache_dir: Path):
    """Load tokenizer + full fp16 model weights natively (no AirLLM, no quantization)."""
    logger.info("Loading tokenizer for %s ...", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, cache_dir=cache_dir)

    logger.info("Loading full fp16 weights for %s into RAM (device_map='cpu') ...", model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, token=hf_token, cache_dir=cache_dir, torch_dtype=torch.float16, device_map="cpu"
    )
    return tokenizer, model


def run_generation(model, tokenizer, prompt: str, max_new_tokens: int) -> dict:
    """Generate a response while measuring TTFT and TPOT via streamed tokens."""
    messages = [{"role": "user", "content": prompt}]
    text = (
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if getattr(tokenizer, "chat_template", None)
        else prompt
    )
    inputs = tokenizer(text, return_tensors="pt")
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    thread = threading.Thread(target=model.generate, kwargs=dict(**inputs, streamer=streamer, max_new_tokens=max_new_tokens))

    start = time.perf_counter()
    thread.start()
    first_token_time, token_count, output_text = None, 0, ""
    for chunk in streamer:
        if first_token_time is None:
            first_token_time = time.perf_counter()
        output_text += chunk
        token_count += 1
    thread.join()
    end = time.perf_counter()

    if first_token_time is None:
        raise RuntimeError("Generation produced no output tokens.")
    return {
        "prompt": prompt,
        "output_text": output_text,
        "output_tokens": token_count,
        "ttft_s": first_token_time - start,
        "tpot_s": (end - first_token_time) / (token_count - 1) if token_count > 1 else None,
    }


def peak_rss_mb() -> float:
    """Peak resident set size for this process, in MB (`ru_maxrss` is in KB on Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def write_result(results_dir: Path, result: dict) -> None:
    out_path = results_dir / "baseline_metrics.json"
    out_path.write_text(json.dumps(result, indent=2))
    logger.info("Result written to %s", out_path)


def main() -> int:
    settings = get_settings()
    settings.ensure_directories()
    configure_logging(settings.logs_dir)
    logger.info("=== ZeroVRAM-AirLLM-Engine: Native Baseline (no AirLLM) ===")
    logger.info("Target model: %s (expected to crawl or OOM on 16GB RAM / 128MB VRAM hardware)", settings.model_id)

    result: dict = {"model_id": settings.model_id}
    hf_token = settings.hugging_face_token.get_secret_value()
    try:
        tokenizer, model = load_model_and_tokenizer(settings.model_id, hf_token, settings.hf_cache_dir)
    except MemoryError:
        logger.error("MemoryError while loading model weights — RAM exhausted.\n%s", traceback.format_exc())
        result["status"] = "oom_at_load"
        write_result(settings.results_dir, result)
        return 137
    except RuntimeError as exc:
        logger.error("RuntimeError while loading model (likely OOM allocator failure): %s", exc)
        result.update(status="runtime_error_at_load", error=str(exc))
        write_result(settings.results_dir, result)
        return 1
    except KeyboardInterrupt:
        logger.warning("Interrupted by user during model load.")
        result["status"] = "interrupted_at_load"
        write_result(settings.results_dir, result)
        return 130

    logger.info("Model loaded. Peak RSS so far: %.0f MB. Attempting generation ...", peak_rss_mb())
    try:
        metrics = run_generation(model, tokenizer, PROMPT, settings.max_new_tokens)
        result.update(metrics, status="success", peak_rss_mb=peak_rss_mb())
        logger.info(
            "SUCCESS — TTFT=%.2fs TPOT=%s tokens=%d peak_rss=%.0fMB",
            metrics["ttft_s"],
            f"{metrics['tpot_s']:.2f}s/token" if metrics["tpot_s"] else "n/a",
            metrics["output_tokens"],
            result["peak_rss_mb"],
        )
        write_result(settings.results_dir, result)
        return 0
    except (MemoryError, RuntimeError) as exc:
        logger.error("Generation failed — memory-bound bottleneck confirmed.\n%s", traceback.format_exc())
        result.update(status="oom_at_generate", error=str(exc), peak_rss_mb=peak_rss_mb())
        write_result(settings.results_dir, result)
        return 137
    except KeyboardInterrupt:
        logger.warning("Interrupted by user during generation.")
        result["status"] = "interrupted_at_generate"
        write_result(settings.results_dir, result)
        return 130


if __name__ == "__main__":
    sys.exit(main())
