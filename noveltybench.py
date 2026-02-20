"""NoveltyBench metrics: diversity (distinct_k) and quality (utility_k).

Wraps the inspect_evals implementation of NoveltyBench (Zhang et al., 2024)
in synchronous helpers suitable for a sweep loop.

Pipeline per prompt:
  1. Partition N responses into functional equivalence classes
     using a fine-tuned DeBERTa-v3-large classifier.
  2. Score representative responses with a Skywork reward model
     (maps raw reward -> discrete 1-10 quality score).
  3. Compute metrics:
       distinct_k  – number of unique equivalence classes
       utility_k   – patience-weighted quality over novel generations
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from inspect_evals.novelty_bench.partition import (
    _load_deberta_classifier,
    _maybe_test_equality,
)
from inspect_evals.novelty_bench.score import (
    _format_scores,
    _identify_class_representatives,
    _load_reward_model,
    _transform_raw_reward,
)

# ---------------------------------------------------------------------------
# Batched DeBERTa pairwise scoring
# ---------------------------------------------------------------------------


def _batch_classifier_scores(
    pairs: list[tuple[str, str]],
    device: str,
    batch_size: int = 64,
) -> list[float]:
    """Score semantic equivalence for many (resp_a, resp_b) pairs in batched
    DeBERTa forward passes, instead of one-at-a-time."""
    if not pairs:
        return []

    model, tokenizer = _load_deberta_classifier(device)
    all_scores: list[float] = []

    for chunk_start in range(0, len(pairs), batch_size):
        chunk = pairs[chunk_start : chunk_start + batch_size]

        raw_iids: list[list[int]] = []
        raw_tids: list[list[int]] = []

        for resp_a, resp_b in chunk:
            iids: list[int] = [tokenizer.cls_token_id]
            for s in [resp_a, resp_b]:
                iids.extend(
                    tokenizer.encode(
                        s, truncation=True, max_length=128, add_special_tokens=False
                    )
                )
                iids.append(tokenizer.sep_token_id)

            seg_boundary = iids.index(tokenizer.sep_token_id) + 1
            tids = [0] * seg_boundary + [1] * (len(iids) - seg_boundary)

            raw_iids.append(iids)
            raw_tids.append(tids)

        max_len = max(len(x) for x in raw_iids)
        pad_id = tokenizer.pad_token_id or 0

        padded_iids = [x + [pad_id] * (max_len - len(x)) for x in raw_iids]
        padded_tids = [x + [0] * (max_len - len(x)) for x in raw_tids]
        attn_mask = [[1] * len(x) + [0] * (max_len - len(x)) for x in raw_iids]

        t_iids = torch.tensor(padded_iids, device=device, dtype=torch.int64)
        t_tids = torch.tensor(padded_tids, device=device, dtype=torch.int64)
        t_mask = torch.tensor(attn_mask, device=device, dtype=torch.int64)

        with torch.inference_mode():
            outputs = model(
                input_ids=t_iids, token_type_ids=t_tids, attention_mask=t_mask
            )
            scores = outputs["logits"].softmax(-1)[:, 1].tolist()

        all_scores.extend(scores)

    return all_scores


# ---------------------------------------------------------------------------
# Batched reward-model scoring
# ---------------------------------------------------------------------------


def _batch_reward_inference(
    conversations: list[list[dict[str, str]]],
    quality_model: str,
    device: str,
    batch_size: int = 32,
) -> list[float]:
    """Run reward inference on all conversations in batches."""
    if not conversations:
        return []

    model, tokenizer = _load_reward_model(quality_model, device)
    all_rewards: list[float] = []

    for i in range(0, len(conversations), batch_size):
        batch = conversations[i : i + batch_size]
        inputs: dict[str, Any] = tokenizer.apply_chat_template(
            batch,
            tokenize=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
            rewards = outputs.logits[:, 0].tolist()
        if isinstance(rewards, float):
            rewards = [rewards]
        all_rewards.extend(rewards)

    return all_rewards


# ---------------------------------------------------------------------------
# Partition helpers (precomputed pairwise scores → greedy clustering)
# ---------------------------------------------------------------------------


def _partition_from_scores(
    n: int,
    pairwise: dict[tuple[int, int], float | bool],
    threshold: float,
) -> list[int]:
    """Greedy sequential partition using precomputed pairwise scores.

    Faithfully reproduces the original algorithm: each response is compared
    against existing class representatives.  _maybe_test_equality False means
    "skip this rep, try the next"; True or classifier > threshold means assign.
    """
    eq_classes: list[int] = []
    reps: list[int] = []

    for i in range(n):
        assigned = False
        for cls, rep in enumerate(reps):
            key = (i, rep) if i > rep else (rep, i)
            val = pairwise[key]

            if isinstance(val, bool):
                if val:
                    eq_classes.append(cls)
                    assigned = True
                    break
                continue

            if val > threshold:
                eq_classes.append(cls)
                assigned = True
                break

        if not assigned:
            eq_classes.append(len(reps))
            reps.append(i)

    return eq_classes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_metrics(
    prompts: list[str],
    responses: list[list[str]],
    device: str = "cuda:0",
    patience: float = 0.8,
    equivalence_threshold: float = 0.102,
    quality_model: str = "small",
    classifier_batch_size: int = 64,
    reward_batch_size: int = 32,
) -> dict[str, torch.Tensor]:
    """Compute all NoveltyBench metrics per prompt (batched internally).

    Args:
        prompts: Q prompt strings.
        responses: Q x N decoded text responses.
        device: torch device for scoring models.
        patience: geometric decay for utility_k (default 0.8).
        equivalence_threshold: DeBERTa classifier threshold (default 0.102).
        quality_model: "small" (Qwen-1.7B) or "large" (Gemma-27B).
        classifier_batch_size: pairs per DeBERTa forward pass.
        reward_batch_size: conversations per reward-model forward pass.

    Returns:
        Dict of (Q,) tensors keyed by metric name:
          distinct_k          – number of unique equivalence classes
          utility_k           – patience-weighted quality over novel generations
          class_quality_mean  – mean reward-model score across unique classes
    """
    _load_deberta_classifier(device)
    _load_reward_model(quality_model, device)

    # --- Phase 1: precompute all pairwise similarities (batched DeBERTa) ---
    needs_classifier: list[tuple[int, int, int]] = []
    per_prompt_pw: list[dict[tuple[int, int], float | bool]] = []

    for q, resps in enumerate(responses):
        pw: dict[tuple[int, int], float | bool] = {}
        n = len(resps)
        for i in range(n):
            for j in range(i):
                eq = _maybe_test_equality(resps[i], resps[j])
                if eq is not None:
                    pw[(i, j)] = eq
                else:
                    needs_classifier.append((q, i, j))
                    pw[(i, j)] = 0.0  # placeholder
        per_prompt_pw.append(pw)

    if needs_classifier:
        text_pairs = [
            (responses[q][i], responses[q][j]) for q, i, j in needs_classifier
        ]
        scores = _batch_classifier_scores(text_pairs, device, classifier_batch_size)
        for (q, i, j), score in zip(needs_classifier, scores):
            per_prompt_pw[q][(i, j)] = score

    # --- Phase 2: partition using precomputed scores ---
    eq_classes_list = [
        _partition_from_scores(len(resps), pw, equivalence_threshold)
        for resps, pw in zip(responses, per_prompt_pw)
    ]

    # --- Phase 3: batch reward scoring across all prompts ---
    all_convs: list[list[dict[str, str]]] = []
    rep_indices_list: list[list[int]] = []

    for q, (prompt, resps, eq_cls) in enumerate(
        zip(prompts, responses, eq_classes_list)
    ):
        rep_idxs = _identify_class_representatives(eq_cls)
        rep_indices_list.append(rep_idxs)
        for ri in rep_idxs:
            all_convs.append(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": resps[ri]},
                ]
            )

    all_raw_rewards = _batch_reward_inference(
        all_convs, quality_model, device, reward_batch_size
    )

    # --- Phase 4: distribute scores and compute metrics ---
    distinct_k: list[float] = []
    utility_k: list[float] = []
    class_quality_mean: list[float] = []
    offset = 0

    for eq_cls, rep_idxs in zip(eq_classes_list, rep_indices_list):
        n_reps = len(rep_idxs)
        raw = all_raw_rewards[offset : offset + n_reps]
        offset += n_reps

        scaled = [_transform_raw_reward(r, quality_model) for r in raw]
        gen_scores, cls_scores = _format_scores(eq_cls, rep_idxs, scaled)

        distinct = max(eq_cls) + 1 if eq_cls else 0
        distinct_k.append(float(distinct))

        if gen_scores:
            utility = float(
                np.average(gen_scores, weights=patience ** np.arange(len(eq_cls)))
            )
        else:
            utility = 0.0
        utility_k.append(utility)

        class_quality_mean.append(float(np.mean(cls_scores)) if cls_scores else 0.0)

    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return {
        "distinct_k": torch.tensor(distinct_k),
        "utility_k": torch.tensor(utility_k),
        "class_quality_mean": torch.tensor(class_quality_mean),
    }


def score_results(
    results_df: pl.DataFrame,
    *,
    device: str = "cuda:0",
    patience: float = 0.8,
    equivalence_threshold: float = 0.102,
    quality_model: str = "small",
    classifier_batch_size: int = 128,
    reward_batch_size: int = 128,
) -> pl.DataFrame:
    """Score a tidy results DataFrame, adding per-prompt metric columns.

    Expects a DataFrame with columns ``prompt_id`` (int), ``prompt`` (str),
    ``sample_id`` (int), ``response`` (str) — as produced by the generation
    methods.  Returns the same DataFrame with three additional columns:
    ``distinct_k``, ``utility_k``, ``class_quality_mean``.
    """
    sorted_df = results_df.sort(["prompt_id", "sample_id"])

    prompt_order = sorted_df.unique("prompt_id", maintain_order=True)
    prompts = prompt_order["prompt"].to_list()
    Q = len(prompts)

    n_samples = int(sorted_df["sample_id"].max()) + 1
    flat = sorted_df["response"].to_list()
    responses = [flat[q * n_samples : (q + 1) * n_samples] for q in range(Q)]

    metrics = compute_metrics(
        prompts,
        responses,
        device=device,
        patience=patience,
        equivalence_threshold=equivalence_threshold,
        quality_model=quality_model,
        classifier_batch_size=classifier_batch_size,
        reward_batch_size=reward_batch_size,
    )

    n = n_samples
    dk = metrics["distinct_k"].mean().item()
    uk = metrics["utility_k"].mean().item()
    print(f"distinct_k={dk:.2f}/{n}  utility_k={uk:.2f}")

    metrics_df = pl.DataFrame(
        {
            "prompt_id": list(range(Q)),
            "distinct_k": metrics["distinct_k"].tolist(),
            "utility_k": metrics["utility_k"].tolist(),
            "class_quality_mean": metrics["class_quality_mean"].tolist(),
        }
    )

    return results_df.join(metrics_df, on="prompt_id")


def benchmark(
    prompts: list[str],
    prompt_ids: list[str],
    runs: list[dict[str, Any]],
    *,
    device: str = "cuda:0",
    patience: float = 0.8,
    equivalence_threshold: float = 0.102,
    quality_model: str = "small",
    classifier_batch_size: int = 128,
    reward_batch_size: int = 128,
) -> pl.DataFrame:
    """Evaluate NoveltyBench metrics across multiple experimental conditions
    in a single batched pass.

    All DeBERTa pairwise comparisons and all reward-model scorings across
    every run are collected and executed in large batches, avoiding the
    overhead of thousands of individual forward passes.

    Args:
        prompts: Q prompt strings.
        prompt_ids: Q prompt identifiers.
        runs: List of dicts, each with keys:
            ``method`` (str), ``temperature`` (float),
            ``responses`` (Q x N list of decoded strings).
        device: torch device for scoring models.
        patience: geometric decay for utility_k.
        equivalence_threshold: DeBERTa classifier threshold.
        quality_model: "small" or "large".
        classifier_batch_size: pairs per DeBERTa forward pass.
        reward_batch_size: conversations per reward-model forward pass.

    Returns:
        polars DataFrame with columns: method, temperature, prompt_id,
        distinct_k, utility_k, class_quality_mean.
    """
    Q = len(prompts)
    n = len(runs[0]["responses"][0]) if runs else 0

    _load_deberta_classifier(device)
    _load_reward_model(quality_model, device)

    # ==== Phase 1: collect ALL pairwise comparison needs ====
    needs_classifier: list[tuple[int, int, int, int]] = []  # (run, q, i, j)
    # pairwise[run][q][(i,j)] = bool | float
    pairwise: list[list[dict[tuple[int, int], float | bool]]] = []

    for r_idx, run in enumerate(runs):
        run_pw: list[dict[tuple[int, int], float | bool]] = []
        for q_idx, resps in enumerate(run["responses"]):
            pw: dict[tuple[int, int], float | bool] = {}
            nr = len(resps)
            for i in range(nr):
                for j in range(i):
                    eq = _maybe_test_equality(resps[i], resps[j])
                    if eq is not None:
                        pw[(i, j)] = eq
                    else:
                        needs_classifier.append((r_idx, q_idx, i, j))
                        pw[(i, j)] = 0.0
            run_pw.append(pw)
        pairwise.append(run_pw)

    # ==== Phase 2: batched DeBERTa scoring ====
    if needs_classifier:
        text_pairs = [
            (runs[r]["responses"][q][i], runs[r]["responses"][q][j])
            for r, q, i, j in needs_classifier
        ]
        scores = _batch_classifier_scores(text_pairs, device, classifier_batch_size)
        for (r, q, i, j), score in zip(needs_classifier, scores):
            pairwise[r][q][(i, j)] = score

    # ==== Phase 3: partition using precomputed scores ====
    eq_all: list[list[list[int]]] = []
    for r_idx, run in enumerate(runs):
        eq_all.append(
            [
                _partition_from_scores(
                    len(resps), pairwise[r_idx][q], equivalence_threshold
                )
                for q, resps in enumerate(run["responses"])
            ]
        )

    # ==== Phase 4: batched reward scoring ====
    all_convs: list[list[dict[str, str]]] = []
    rep_all: list[list[list[int]]] = []

    for r_idx, run in enumerate(runs):
        run_reps: list[list[int]] = []
        for q_idx in range(Q):
            rep_idxs = _identify_class_representatives(eq_all[r_idx][q_idx])
            run_reps.append(rep_idxs)
            resps = run["responses"][q_idx]
            for ri in rep_idxs:
                all_convs.append(
                    [
                        {"role": "user", "content": prompts[q_idx]},
                        {"role": "assistant", "content": resps[ri]},
                    ]
                )
        rep_all.append(run_reps)

    all_raw_rewards = _batch_reward_inference(
        all_convs, quality_model, device, reward_batch_size
    )

    # ==== Phase 5: distribute scores & compute metrics ====
    rows: list[dict[str, Any]] = []
    offset = 0

    for r_idx, run in enumerate(runs):
        for q_idx in range(Q):
            eq_cls = eq_all[r_idx][q_idx]
            rep_idxs = rep_all[r_idx][q_idx]
            n_reps = len(rep_idxs)
            raw = all_raw_rewards[offset : offset + n_reps]
            offset += n_reps

            scaled = [_transform_raw_reward(r, quality_model) for r in raw]
            gen_scores, cls_scores = _format_scores(eq_cls, rep_idxs, scaled)

            distinct = max(eq_cls) + 1 if eq_cls else 0
            nr = len(eq_cls)
            utility = (
                float(np.average(gen_scores, weights=patience ** np.arange(nr)))
                if gen_scores
                else 0.0
            )
            cqm = float(np.mean(cls_scores)) if cls_scores else 0.0

            rows.append(
                {
                    "method": run["method"],
                    "temperature": run["temperature"],
                    "prompt_id": prompt_ids[q_idx],
                    "distinct_k": distinct,
                    "utility_k": utility,
                    "class_quality_mean": cqm,
                }
            )

        run_rows = rows[-Q:]
        dk = np.mean([r["distinct_k"] for r in run_rows])
        uk = np.mean([r["utility_k"] for r in run_rows])
        print(
            f"{run['method']} temp={run['temperature']:.2f}: "
            f"distinct_k={dk:.2f}/{n}  utility_k={uk:.2f}"
        )

    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_COLORS = ["#55A868", "#4C72B0", "#DD8452", "#8172B3", "#937860", "#DA8BC3"]
_MARKERS = ["D", "s", "o", "^", "v", "P"]

CATEGORY_GROUPS: dict[str, str] = {
    "Creativity": "Creative",
    "Character & Entity Naming": "Creative",
    "Factual Knowledge": "Factual",
    "Random Generation & Selection": "Randomness",
    "Subjective Rankings & Opinions": "Subjectivity",
    "Product & Purchase Recommendations": "Subjectivity",
}

CATEGORY_ORDER: list[str] = ["Creative", "Factual", "Randomness", "Subjectivity"]


def plot_metrics(
    methods: dict[str, pl.DataFrame],
    *,
    n: int = 8,
    human_baselines: dict[str, dict[str, float]] | None = None,
    title: str | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Four-row × two-column NoveltyBench plot (rows = category, cols = metric).

    Args:
        methods: ``{display_name: DataFrame}``.  Each DataFrame must contain
            columns ``temperature_response``, ``category``, ``distinct_k``,
            and ``utility_k``.  Raw fine-grained categories are mapped to
            the four high-level groups via :data:`CATEGORY_GROUPS`.
        n: Number of samples per prompt (controls the y-axis *max* line).
        human_baselines: Optional ``{category_group: {"distinct_k": v,
            "utility_k": v}}``.
        title: Figure suptitle.
        save_path: If given, save the figure to this path.

    Returns:
        The matplotlib Figure.
    """
    n_cats = len(CATEGORY_ORDER)
    fig, axes = plt.subplots(n_cats, 2, figsize=(12, 3.8 * n_cats), squeeze=False)

    for i, (name, df) in enumerate(methods.items()):
        color = _COLORS[i % len(_COLORS)]
        marker = _MARKERS[i % len(_MARKERS)]

        if "category_group" not in df.columns:
            cat_col = [CATEGORY_GROUPS.get(c, c) for c in df["category"].to_list()]
            df = df.with_columns(pl.Series("category_group", cat_col))

        for row, cat in enumerate(CATEGORY_ORDER):
            cat_df = df.filter(pl.col("category_group") == cat)
            if cat_df.is_empty():
                continue

            stats = (
                cat_df.group_by("temperature_response")
                .agg(
                    pl.col("utility_k").mean().alias("uk_mean"),
                    (
                        pl.col("utility_k").std() / pl.col("utility_k").count().sqrt()
                    ).alias("uk_sem"),
                    pl.col("distinct_k").mean().alias("dk_mean"),
                    (
                        pl.col("distinct_k").std() / pl.col("distinct_k").count().sqrt()
                    ).alias("dk_sem"),
                )
                .sort("temperature_response")
            )

            x = stats["temperature_response"].to_numpy()

            ax_util = axes[row, 0]
            ax_div = axes[row, 1]

            for ax, mean_col, sem_col in [
                (ax_util, "uk_mean", "uk_sem"),
                (ax_div, "dk_mean", "dk_sem"),
            ]:
                y = stats[mean_col].to_numpy()
                err = stats[sem_col].to_numpy()
                ax.plot(x, y, f"{marker}-", color=color, ms=5, lw=1.5, label=name)
                ax.fill_between(x, y - err, y + err, alpha=0.15, color=color)

    for row, cat in enumerate(CATEGORY_ORDER):
        ax_util = axes[row, 0]
        ax_div = axes[row, 1]

        ax_div.axhline(n, color="gray", ls="--", lw=0.8, label=f"max = {n}")
        ax_div.set_ylim(0, n + 0.5)

        if human_baselines and cat in human_baselines:
            hb = human_baselines[cat]
            ax_div.axhline(
                hb["distinct_k"],
                color="#C44E52",
                ls="--",
                lw=1.0,
                label=f"human = {hb['distinct_k']:.1f}",
            )
            ax_util.axhline(
                hb["utility_k"],
                color="#C44E52",
                ls="--",
                lw=1.0,
                label=f"human = {hb['utility_k']:.1f}",
            )

        ax_util.set_ylabel(cat, fontsize=11, fontweight="bold")
        ax_div.set_ylabel("")

        for ax in (ax_util, ax_div):
            ax.spines[["top", "right"]].set_visible(False)
            handles, labels = ax.get_legend_handles_labels()
            seen: dict[str, int] = {}
            for idx, lbl in enumerate(labels):
                seen.setdefault(lbl, idx)
            deduped = [(handles[v], k) for k, v in seen.items()]
            ax.legend(
                [h for h, _ in deduped],
                [lbl for _, lbl in deduped],
                frameon=False,
                fontsize=7,
            )

        if row == 0:
            ax_util.set_title("utility_k  (diversity × quality)")
            ax_div.set_title(f"distinct_k  (diversity, max {n})")
        if row == n_cats - 1:
            ax_util.set_xlabel("temperature_response")
            ax_div.set_xlabel("temperature_response")

    if title:
        fig.suptitle(title, fontsize=11, y=1.01)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
