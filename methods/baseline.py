"""Baseline method: sample N responses at a fixed temperature."""

from __future__ import annotations

import polars as pl

from utils import build_tidy_results, check_cache, generate_batched


def generate(
    model,
    tokenizer,
    prompts: list[str],
    n_samples: int,
    *,
    model_name: str,
    temperature: float,
    max_new_tokens: int = 512,
    batch_size: int = 128,
    use_cache: bool = True,
    results_dir: str = "results",
) -> tuple[pl.DataFrame, dict]:
    """Generate N responses per prompt by direct sampling.

    Returns (results_df, metadata).
    """
    metadata = {
        "id": f"baseline_t={temperature}",
        "name": "baseline",
        "model": model_name,
        "n_samples": n_samples,
        "temperature_response": temperature,
        "max_new_tokens": max_new_tokens,
    }

    if use_cache:
        cached = check_cache(metadata, results_dir)
        if cached is not None:
            print(f"  [cached] baseline  temp={temperature}")
            return cached, metadata

    print(f"  Generating baseline  temp={temperature} ...", end=" ", flush=True)

    flat_prompts = [q for q in prompts for _ in range(n_samples)]
    flat_responses = generate_batched(
        flat_prompts,
        model,
        tokenizer,
        batch_size,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
    )
    responses = [
        flat_responses[q * n_samples : (q + 1) * n_samples] for q in range(len(prompts))
    ]

    print("done")
    return build_tidy_results(prompts, responses), metadata
