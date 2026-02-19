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

import matplotlib.pyplot as plt
import numpy as np
import torch
from inspect_evals.novelty_bench.partition import (
    _classifier_score,
    _load_deberta_classifier,
    _maybe_test_equality,
)
from inspect_evals.novelty_bench.score import (
    _format_scores,
    _identify_class_representatives,
    _load_reward_model,
    _run_reward_inference,
    _transform_raw_reward,
)


def _partition_responses(
    responses: list[str],
    equivalence_threshold: float = 0.102,
    device: str = "cpu",
) -> list[int]:
    """Partition responses into functional equivalence classes (synchronous)."""
    equivalence_classes: list[int] = []
    representatives: list[str] = []

    for response in responses:
        assigned = False
        for class_idx, representative in enumerate(representatives):
            equality = _maybe_test_equality(response, representative)
            if equality is not None:
                if equality:
                    equivalence_classes.append(class_idx)
                    assigned = True
                    break
                continue

            score = _classifier_score(response, representative, device)
            if score > equivalence_threshold:
                equivalence_classes.append(class_idx)
                assigned = True
                break

        if not assigned:
            equivalence_classes.append(len(representatives))
            representatives.append(response)

    return equivalence_classes


def _score_representatives(
    prompt: str,
    generations: list[str],
    equivalence_classes: list[int],
    quality_model: str = "small",
    device: str = "cpu",
) -> tuple[list[float], list[float]]:
    """Score each equivalence class representative with a reward model (synchronous).

    Returns (generation_scores, class_scores):
      generation_scores – one score per generation; duplicates get 0
      class_scores      – one score per unique equivalence class
    """
    model, tokenizer = _load_reward_model(quality_model, device)

    representative_indices = _identify_class_representatives(equivalence_classes)
    representative_generations = [generations[i] for i in representative_indices]

    raw_rewards = _run_reward_inference(
        model, tokenizer, prompt, representative_generations, device
    )
    scaled_scores = [_transform_raw_reward(r, quality_model) for r in raw_rewards]

    return _format_scores(equivalence_classes, representative_indices, scaled_scores)


def compute_metrics(
    prompts: list[str],
    responses: list[list[str]],
    device: str = "cuda:0",
    patience: float = 0.8,
    equivalence_threshold: float = 0.102,
    quality_model: str = "small",
) -> dict[str, torch.Tensor]:
    """Compute all NoveltyBench metrics per prompt.

    Args:
        prompts: Q prompt strings.
        responses: Q x N decoded text responses.
        device: torch device for scoring models.
        patience: geometric decay for utility_k (default 0.8).
        equivalence_threshold: DeBERTa classifier threshold (default 0.102).
        quality_model: "small" (Qwen-1.7B) or "large" (Gemma-27B).

    Returns:
        Dict of (Q,) tensors keyed by metric name:
          distinct_k          – number of unique equivalence classes
          utility_k           – patience-weighted quality over novel generations
          class_quality_mean  – mean reward-model score across unique classes
    """
    _load_deberta_classifier(device)
    _load_reward_model(quality_model, device)

    distinct_k: list[float] = []
    utility_k: list[float] = []
    class_quality_mean: list[float] = []

    for prompt, resps in zip(prompts, responses):
        eq_classes = _partition_responses(resps, equivalence_threshold, device)

        gen_scores, cls_scores = _score_representatives(
            prompt, resps, eq_classes, quality_model, device
        )

        distinct = max(eq_classes) + 1 if eq_classes else 0
        distinct_k.append(float(distinct))

        if gen_scores:
            utility = float(
                np.average(gen_scores, weights=patience ** np.arange(len(resps)))
            )
        else:
            utility = 0.0
        utility_k.append(utility)

        class_quality_mean.append(float(np.mean(cls_scores)) if cls_scores else 0.0)

    return {
        "distinct_k": torch.tensor(distinct_k),
        "utility_k": torch.tensor(utility_k),
        "class_quality_mean": torch.tensor(class_quality_mean),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_COLORS = ["#55A868", "#4C72B0", "#DD8452", "#8172B3", "#937860", "#DA8BC3"]
_MARKERS = ["D", "s", "o", "^", "v", "P"]


def plot_metrics(
    methods: dict[str, dict[str, np.ndarray]],
    *,
    n: int = 8,
    human_distinct_k: float | None = None,
    human_utility_k: float | None = None,
    xlabel: str = "Temperature",
    title: str | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Two-panel NoveltyBench plot with one line per method.

    Args:
        methods: Mapping of method name to results dict. Each results dict
            must contain the keys ``x``, ``distinct_k_mean``,
            ``distinct_k_sem``, ``utility_k_mean``, and ``utility_k_sem``
            (all 1-D numpy arrays of the same length).
        n: Number of samples per prompt (controls the y-axis ``max`` line).
        human_distinct_k: If given, draw a human-baseline line on the
            distinct_k panel.
        human_utility_k: If given, draw a human-baseline line on the
            utility_k panel.
        xlabel: Shared x-axis label.
        title: Figure suptitle.
        save_path: If given, save the figure to this path.

    Returns:
        The matplotlib Figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for i, (name, r) in enumerate(methods.items()):
        color = _COLORS[i % len(_COLORS)]
        marker = _MARKERS[i % len(_MARKERS)]
        x = r["x"]

        ax1.plot(
            x,
            r["distinct_k_mean"],
            f"{marker}-",
            color=color,
            ms=5,
            lw=1.5,
            label=name,
        )
        ax1.fill_between(
            x,
            r["distinct_k_mean"] - r["distinct_k_sem"],
            r["distinct_k_mean"] + r["distinct_k_sem"],
            alpha=0.15,
            color=color,
        )

        ax2.plot(
            x,
            r["utility_k_mean"],
            f"{marker}-",
            color=color,
            ms=5,
            lw=1.5,
            label=name,
        )
        ax2.fill_between(
            x,
            r["utility_k_mean"] - r["utility_k_sem"],
            r["utility_k_mean"] + r["utility_k_sem"],
            alpha=0.15,
            color=color,
        )

    ax1.axhline(n, color="gray", ls="--", lw=0.8, label=f"max = {n}")
    if human_distinct_k is not None:
        ax1.axhline(
            human_distinct_k,
            color="#C44E52",
            ls="--",
            lw=1.2,
            label=f"human = {human_distinct_k:.1f}",
        )
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(f"Distinct classes (out of {n})")
    ax1.set_title("distinct_k (diversity)")
    ax1.set_ylim(0, n + 0.5)
    ax1.legend(frameon=False)
    ax1.spines[["top", "right"]].set_visible(False)

    if human_utility_k is not None:
        ax2.axhline(
            human_utility_k,
            color="#C44E52",
            ls="--",
            lw=1.2,
            label=f"human = {human_utility_k:.1f}",
        )
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("Utility score")
    ax2.set_title("utility_k (diversity × quality)")
    ax2.legend(frameon=False)
    ax2.spines[["top", "right"]].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
