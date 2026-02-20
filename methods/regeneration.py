"""Regeneration method: sequential multi-turn sampling in the same conversation.

Instead of sampling N independent responses, this method builds a growing
conversation per prompt — each turn asks for "another different answer" — and
collects the N sequential responses.  Turns are sequential (each depends on
the previous response) but all Q prompts are batched within each turn.
"""

from __future__ import annotations

import polars as pl

from utils import build_tidy_results, check_cache, generate_batched_conversations

DEFAULT_REPETITION_PROMPT = (
    "Now give me another, different answer to the original question."
)


def generate(
    model,
    tokenizer,
    prompts: list[str],
    n_samples: int,
    *,
    model_name: str,
    temperature_response: float,
    max_new_tokens: int = 512,
    repetition_prompt: str = DEFAULT_REPETITION_PROMPT,
    batch_size: int = 128,
    use_cache: bool = True,
    results_dir: str = "results",
) -> tuple[pl.DataFrame, dict]:
    """Generate N responses per prompt by repeatedly prompting in the same conversation.

    Turn 0: [user: prompt]                                         → response_0
    Turn 1: [user: prompt, asst: response_0, user: repetition]     → response_1
    Turn k: [..., asst: response_{k-1}, user: repetition]          → response_k

    Sequential across turns, batched across all Q prompts at each turn.

    Returns (results_df, metadata).
    """
    metadata = {
        "id": f"regeneration_t={temperature_response}",
        "name": "regeneration",
        "model_name": model_name,
        "n_samples": n_samples,
        "temperature_response": temperature_response,
        "max_new_tokens": max_new_tokens,
        "repetition_prompt": repetition_prompt,
    }

    if use_cache:
        cached = check_cache(metadata, results_dir)
        if cached is not None:
            print(f"  [cached] regeneration  t={temperature_response}")
            return cached, metadata

    print(
        f"  Generating regeneration  t={temperature_response} ...",
        end=" ",
        flush=True,
    )

    Q = len(prompts)
    conversations: list[list[dict[str, str]]] = [
        [{"role": "user", "content": p}] for p in prompts
    ]
    responses: list[list[str]] = [[] for _ in range(Q)]

    for turn in range(n_samples):
        turn_responses = generate_batched_conversations(
            conversations,
            model,
            tokenizer,
            batch_size,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature_response,
        )

        for q in range(Q):
            resp = turn_responses[q]
            responses[q].append(resp)
            conversations[q] = conversations[q] + [
                {"role": "assistant", "content": resp},
                {"role": "user", "content": repetition_prompt},
            ]

    print("done")
    return build_tidy_results(prompts, responses), metadata
