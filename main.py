# %%
import json
import urllib.request
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from methods import baseline, ifg, regeneration, steering
from noveltybench import CATEGORY_GROUPS, plot_metrics, score_results
from utils import save_results, seed_everything

load_dotenv()
seed_everything(42)

# %% Configuration
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
N = 8
MAX_NEW_TOKENS = 64
BATCH_SIZE = 128
RESULTS_DIR = "results"

METHOD_MODULES = {
    "baseline": baseline,
    "ifg": ifg,
    "regeneration": regeneration,
    "steering": steering,
}

DISPLAY_LABELS = {
    "baseline": "Baseline",
    "ifg": "IFG",
    "regeneration": "Regeneration",
    "steering": "Steering",
}

_COMMON = {
    "model_name": MODEL_NAME,
    "n_samples": N,
    "max_new_tokens": MAX_NEW_TOKENS,
}

EXPERIMENTS = [
    {**_COMMON, "name": "baseline", "temperature_response": 0.01},
    {**_COMMON, "name": "baseline", "temperature_response": 0.6},
    {**_COMMON, "name": "baseline", "temperature_response": 1.2},
]

# %% Load NB-Curated (100 prompts: randomness, factual, creative, subjectivity)
ds = load_dataset("yimingzhang/novelty-bench", split="curated")
QUESTIONS = [row["prompt"] for row in ds]
QUESTION_IDS = [row["id"] for row in ds]
Q = len(QUESTIONS)
print(f"Loaded {Q} prompts from NB-Curated")

_CURATED_URL = "https://raw.githubusercontent.com/novelty-bench/novelty-bench/main/data/curated.jsonl"
with urllib.request.urlopen(_CURATED_URL) as resp:
    _curated_data = {row["id"]: row for line in resp if (row := json.loads(line))}

QUESTION_CATEGORIES = [_curated_data[qid]["category"] for qid in QUESTION_IDS]

_HUMANS_URL = "https://raw.githubusercontent.com/novelty-bench/novelty-bench/main/data/humans.jsonl"
with urllib.request.urlopen(_HUMANS_URL) as resp:
    _human_data = {
        row["id"]: row
        for line in resp
        if (row := json.loads(line))["id"] in set(QUESTION_IDS)
    }

_human_ids = [qid for qid in QUESTION_IDS if qid in _human_data]
human_distinct_k = np.array([_human_data[qid]["distinct"] for qid in _human_ids])
human_utility_k = np.array([_human_data[qid]["utility"] for qid in _human_ids])
n_human = len(_human_ids)
print(
    f"Human baselines ({n_human} prompts, 8 annotators): "
    f"distinct_k={human_distinct_k.mean():.2f}±{human_distinct_k.std() / np.sqrt(n_human):.2f}, "
    f"utility_k={human_utility_k.mean():.2f}±{human_utility_k.std() / np.sqrt(n_human):.2f}"
)

_human_by_cat: dict[str, dict[str, list[float]]] = {}
for qid in _human_ids:
    cat_group = CATEGORY_GROUPS[_curated_data[qid]["category"]]
    bucket = _human_by_cat.setdefault(cat_group, {"distinct_k": [], "utility_k": []})
    bucket["distinct_k"].append(_human_data[qid]["distinct"])
    bucket["utility_k"].append(_human_data[qid]["utility"])

human_baselines = {
    cat: {
        "distinct_k": float(np.mean(v["distinct_k"])),
        "utility_k": float(np.mean(v["utility_k"])),
    }
    for cat, v in _human_by_cat.items()
}

# %% Load model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16, device_map="cuda:0"
)
model.config.pad_token_id = tokenizer.pad_token_id


# %% Run experiments
def _run_experiment(exp: dict) -> tuple[pl.DataFrame, dict]:
    """Dispatch an experiment dict to the appropriate method module."""
    exp = dict(exp)
    name = exp.pop("name")
    n_samples = exp.pop("n_samples")
    module = METHOD_MODULES[name]
    return module.generate(
        model,
        tokenizer,
        QUESTIONS,
        n_samples,
        batch_size=BATCH_SIZE,
        results_dir=RESULTS_DIR,
        **exp,
    )


def _finalize(results_df: pl.DataFrame, meta: dict) -> pl.DataFrame:
    """Add category column, score, and cache.  No-op if already scored (cache hit)."""
    if "distinct_k" in results_df.columns:
        return results_df

    results_df = results_df.with_columns(
        pl.Series(
            "category",
            [QUESTION_CATEGORIES[pid] for pid in results_df["prompt_id"].to_list()],
        )
    )
    print("  Scoring ... ", end="", flush=True)
    results_df = score_results(results_df, device="cuda:0")
    save_results(results_df, meta, RESULTS_DIR)
    return results_df


def _extract_metrics(results_df: pl.DataFrame, meta: dict) -> pl.DataFrame:
    """Extract per-prompt metrics from a scored tidy DataFrame."""
    return (
        results_df.unique("prompt_id", maintain_order=True)
        .select(
            "prompt_id", "category", "distinct_k", "utility_k", "class_quality_mean"
        )
        .with_columns(
            pl.lit(meta["temperature_response"]).alias("temperature_response")
        )
    )


method_dfs: dict[str, list[pl.DataFrame]] = {}

for exp in EXPERIMENTS:
    results_df, meta = _run_experiment(exp)
    results_df = _finalize(results_df, meta)
    label = DISPLAY_LABELS[meta["name"]]
    method_dfs.setdefault(label, []).append(_extract_metrics(results_df, meta))

# %% Build per-method DataFrames and combined CSV
methods_combined: dict[str, pl.DataFrame] = {
    name: pl.concat(dfs) for name, dfs in method_dfs.items()
}

all_df = pl.concat(
    df.with_columns(pl.lit(name).alias("method"))
    for name, df in methods_combined.items()
)
csv_path = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
all_df.write_csv(csv_path)
print(f"\nSaved {len(all_df)} rows to {csv_path}")
print(all_df)

# %% Plot
fig = plot_metrics(
    methods_combined,
    n=N,
    human_baselines=human_baselines,
    title=f"{MODEL_NAME}  ·  N={N}  ·  {Q} NB-Curated prompts  ·  ±1 SEM",
    save_path=f"{RESULTS_DIR}/plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
)
plt.show()
