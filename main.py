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

from noveltybench import compute_metrics, plot_metrics

load_dotenv()

# %% Configuration
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
N = 8
MAX_NEW_TOKENS = 512  # NoveltyBench paper default
BATCH_SIZE = 32
TEMPERATURES = [0.01, 0.4, 0.8, 1.2]
T_RESPONSE_IFG = 0.6

# %% Load NB-Curated (100 prompts: randomness, factual, creative, subjectivity)
ds = load_dataset("yimingzhang/novelty-bench", split="curated")
QUESTIONS = [row["prompt"] for row in ds]
QUESTION_IDS = [row["id"] for row in ds]
Q = len(QUESTIONS)
print(f"Loaded {Q} prompts from NB-Curated")

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

# %% Load model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16, device_map="cuda:0"
)
model.config.pad_token_id = tokenizer.pad_token_id


# %% Generation helpers
def _tokenize(prompts: list[str]):
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
    return tokenizer(chat_texts, return_tensors="pt", padding=True).to(model.device)


@torch.no_grad()
def generate_batched(prompts: list[str], **kwargs) -> list[str]:
    """Generate one response per prompt, in BATCH_SIZE chunks. Returns decoded text."""
    results = []
    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i : i + BATCH_SIZE]
        inputs = _tokenize(batch)
        output_ids = model.generate(**inputs, **kwargs)
        generated = output_ids[:, inputs["input_ids"].shape[1] :]
        results.extend(
            tokenizer.decode(generated[j], skip_special_tokens=True).strip()
            for j in range(generated.shape[0])
        )
    return results


def generate_responses(questions: list[str], n: int, **kwargs) -> list[list[str]]:
    """Generate n responses per question. Returns Q × N list of decoded strings."""
    flat = generate_batched([q for q in questions for _ in range(n)], **kwargs)
    return [flat[q * n : (q + 1) * n] for q in range(len(questions))]


# %% IFG (Intent Factored Generation) — Ahmed et al., 2025
INTENT_INSTRUCTION = (
    "Below is a prompt. Produce a brief direction for a response — "
    "just 3-5 keywords or a single concept sentence. "
    "Output ONLY the direction, nothing else.\n\n"
    "Prompt: {question}\n\nDirection:"
)

IFG_RESPONSE_TEMPLATE = "{question}\n\n[Follow this direction: {intent}]"


def generate_intents(questions: list[str], n: int, **kwargs) -> list[list[str]]:
    """IFG Stage 1: sample N intent strings per question."""
    flat = generate_batched(
        [INTENT_INSTRUCTION.format(question=q) for q in questions for _ in range(n)],
        max_new_tokens=48,
        **kwargs,
    )
    return [flat[q * n : (q + 1) * n] for q in range(len(questions))]


def generate_from_intents(
    questions: list[str], intents: list[list[str]], **kwargs
) -> list[list[str]]:
    """IFG Stage 2: generate responses conditioned on (question, intent) pairs."""
    Q, N_i = len(questions), len(intents[0])
    flat = generate_batched(
        [
            IFG_RESPONSE_TEMPLATE.format(question=questions[q], intent=intents[q][n])
            for q in range(Q)
            for n in range(N_i)
        ],
        **kwargs,
    )
    return [flat[q * N_i : (q + 1) * N_i] for q in range(Q)]


# %% Run experiments
results_rows: list[dict] = []


def _run_and_record(method: str, temperature: float, responses: list[list[str]]):
    nb = compute_metrics(QUESTIONS, responses, device="cuda:0")
    for q_idx in range(Q):
        results_rows.append(
            {
                "method": method,
                "temperature": temperature,
                "prompt_id": QUESTION_IDS[q_idx],
                "distinct_k": nb["distinct_k"][q_idx].item(),
                "utility_k": nb["utility_k"][q_idx].item(),
                "class_quality_mean": nb["class_quality_mean"][q_idx].item(),
            }
        )
    dk = nb["distinct_k"].mean().item()
    uk = nb["utility_k"].mean().item()
    print(f"distinct_k={dk:.2f}/{N}  utility_k={uk:.2f}")


for temp in TEMPERATURES:
    print(f"Baseline temp={temp:.2f} ... ", end="", flush=True)
    responses = generate_responses(
        QUESTIONS,
        N,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=float(temp),
    )
    _run_and_record("baseline", float(temp), responses)

# %%
for t_i in TEMPERATURES:
    print(f"IFG t_intent={t_i:.2f}, t_resp={T_RESPONSE_IFG} ... ", end="", flush=True)
    intents = generate_intents(QUESTIONS, N, do_sample=True, temperature=float(t_i))
    responses = generate_from_intents(
        QUESTIONS,
        intents,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=T_RESPONSE_IFG,
    )
    _run_and_record(f"ifg_t_resp={T_RESPONSE_IFG}", float(t_i), responses)

# %% Save results to timestamped CSV
df = pl.DataFrame(results_rows)
csv_path = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
df.write_csv(csv_path)
print(f"Saved {len(df)} rows to {csv_path}")
print(df)


# %% Plot
def _summarize_for_plot(df: pl.DataFrame, method: str) -> dict:
    s = (
        df.filter(pl.col("method") == method)
        .group_by("temperature")
        .agg(
            pl.col("distinct_k").mean().alias("distinct_k_mean"),
            (pl.col("distinct_k").std() / pl.col("distinct_k").count().sqrt()).alias(
                "distinct_k_sem"
            ),
            pl.col("utility_k").mean().alias("utility_k_mean"),
            (pl.col("utility_k").std() / pl.col("utility_k").count().sqrt()).alias(
                "utility_k_sem"
            ),
        )
        .sort("temperature")
    )
    return {
        "x": s["temperature"].to_numpy(),
        "distinct_k_mean": s["distinct_k_mean"].to_numpy(),
        "distinct_k_sem": s["distinct_k_sem"].to_numpy(),
        "utility_k_mean": s["utility_k_mean"].to_numpy(),
        "utility_k_sem": s["utility_k_sem"].to_numpy(),
    }


ifg_method = f"ifg_t_resp={T_RESPONSE_IFG}"
fig = plot_metrics(
    {
        "Baseline": _summarize_for_plot(df, "baseline"),
        f"IFG ($t_{{resp}}={T_RESPONSE_IFG}$)": _summarize_for_plot(df, ifg_method),
    },
    n=N,
    human_distinct_k=float(human_distinct_k.mean()),
    human_utility_k=float(human_utility_k.mean()),
    xlabel="Temperature",
    title=f"{MODEL_NAME}  ·  N={N}  ·  {Q} NB-Curated prompts  ·  ±1 SEM",
    save_path="diversity_vs_temperature.png",
)
plt.show()

# %%
