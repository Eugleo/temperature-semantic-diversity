"""Shared utilities: caching, generation helpers, result formatting, seeding."""

from __future__ import annotations

import hashlib
import json
import os
import random
from datetime import datetime

import numpy as np
import polars as pl
import torch

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

_CACHE_INDEX = ".cache_index.json"


def _metadata_hash(metadata: dict) -> str:
    return hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[
        :16
    ]


def _load_cache_index(results_dir: str) -> dict[str, str]:
    path = os.path.join(results_dir, _CACHE_INDEX)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_cache_index(results_dir: str, index: dict[str, str]) -> None:
    path = os.path.join(results_dir, _CACHE_INDEX)
    with open(path, "w") as f:
        json.dump(index, f, indent=2)


def check_cache(metadata: dict, results_dir: str = "results") -> pl.DataFrame | None:
    """Return cached results DataFrame if metadata matches, else None."""
    h = _metadata_hash(metadata)
    index = _load_cache_index(results_dir)
    if h not in index:
        return None
    path = os.path.join(results_dir, index[h])
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    if data.get("metadata") == metadata:
        return pl.DataFrame(data["results"])
    return None


def save_results(
    results: pl.DataFrame, metadata: dict, results_dir: str = "results"
) -> str:
    """Write results + metadata as JSON.  Filename uses metadata ``id`` + timestamp."""
    os.makedirs(results_dir, exist_ok=True)
    run_id = metadata.get("id", "unknown")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{run_id}_{timestamp}.json"
    path = os.path.join(results_dir, filename)
    data = {"metadata": metadata, "results": results.to_dicts()}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    h = _metadata_hash(metadata)
    index = _load_cache_index(results_dir)
    index[h] = filename
    _save_cache_index(results_dir, index)
    return path


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------


def tokenize_prompts(prompts: list[str], tokenizer, device: torch.device | str) -> dict:
    """Tokenize single-turn user prompts with left-padding for batch generation."""
    tokenizer.padding_side = "left"
    chat_texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for p in prompts
    ]
    return tokenizer(chat_texts, return_tensors="pt", padding=True).to(device)


def tokenize_conversations(
    conversations: list[list[dict[str, str]]],
    tokenizer,
    device: torch.device | str,
) -> dict:
    """Tokenize multi-turn conversations with left-padding for batch generation."""
    tokenizer.padding_side = "left"
    chat_texts = [
        tokenizer.apply_chat_template(
            conv,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for conv in conversations
    ]
    return tokenizer(chat_texts, return_tensors="pt", padding=True).to(device)


@torch.no_grad()
def generate_batched(
    prompts: list[str],
    model,
    tokenizer,
    batch_size: int,
    **kwargs,
) -> list[str]:
    """Generate one response per single-turn prompt, in batch_size chunks."""
    results: list[str] = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inputs = tokenize_prompts(batch, tokenizer, model.device)
        output_ids = model.generate(**inputs, **kwargs)
        generated = output_ids[:, inputs["input_ids"].shape[1] :]
        results.extend(
            tokenizer.decode(generated[j], skip_special_tokens=True).strip()
            for j in range(generated.shape[0])
        )
    return results


@torch.no_grad()
def generate_continuation_batched(
    formatted_texts: list[str],
    model,
    tokenizer,
    batch_size: int,
    **kwargs,
) -> list[str]:
    """Generate continuations from pre-formatted texts (chat template already applied).

    Unlike generate_batched, this skips chat-template wrapping — the caller
    provides fully formatted text including any special tokens and partial
    assistant content.
    """
    results: list[str] = []
    for i in range(0, len(formatted_texts), batch_size):
        batch = formatted_texts[i : i + batch_size]
        tokenizer.padding_side = "left"
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
        output_ids = model.generate(**inputs, **kwargs)
        generated = output_ids[:, inputs["input_ids"].shape[1] :]
        results.extend(
            tokenizer.decode(generated[j], skip_special_tokens=True).strip()
            for j in range(generated.shape[0])
        )
    return results


@torch.no_grad()
def generate_batched_conversations(
    conversations: list[list[dict[str, str]]],
    model,
    tokenizer,
    batch_size: int,
    **kwargs,
) -> list[str]:
    """Generate one response per multi-turn conversation, in batch_size chunks."""
    results: list[str] = []
    for i in range(0, len(conversations), batch_size):
        batch = conversations[i : i + batch_size]
        inputs = tokenize_conversations(batch, tokenizer, model.device)
        output_ids = model.generate(**inputs, **kwargs)
        generated = output_ids[:, inputs["input_ids"].shape[1] :]
        results.extend(
            tokenizer.decode(generated[j], skip_special_tokens=True).strip()
            for j in range(generated.shape[0])
        )
    return results


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------


def build_tidy_results(prompts: list[str], responses: list[list[str]]) -> pl.DataFrame:
    """Build a tidy DataFrame from Q x N responses.

    Returns DataFrame with columns: prompt_id, prompt, sample_id, response.
    """
    rows = [
        {"prompt_id": q, "prompt": prompt, "sample_id": s, "response": resp}
        for q, (prompt, resps) in enumerate(zip(prompts, responses))
        for s, resp in enumerate(resps)
    ]
    return pl.DataFrame(rows)


def tidy_to_responses(df: pl.DataFrame) -> list[list[str]]:
    """Convert tidy results DataFrame back to Q x N response lists."""
    sorted_df = df.sort(["prompt_id", "sample_id"])
    n_samples = int(sorted_df["sample_id"].max()) + 1
    flat = sorted_df["response"].to_list()
    Q = len(flat) // n_samples
    return [flat[q * n_samples : (q + 1) * n_samples] for q in range(Q)]
