"""IFG (Intent Factored Generation) — Ahmed et al., 2025."""

from __future__ import annotations

import polars as pl

from utils import (
    build_tidy_results,
    check_cache,
    generate_batched,
    generate_continuation_batched,
)

SEPARATOR = "###"

# One-shot prompt following the paper's keyword-intent format (Appendix I).
# The example uses a 3-sentence story so the model learns the expected length
# and keyword→response relationship.
DEFAULT_FEW_SHOT_PROMPT = """\
Below are prompts and responses. Before each response, produce a few \
comma-separated keywords that capture the core concepts the response \
will cover.

### Prompt: Write a three-sentence story about a lost dog finding its way home.
### Keywords:
loyalty, rainstorm, familiar scent, porch light
### Response:
The small terrier shivered under a bus stop bench as rain hammered the \
pavement, three miles from anything she recognized. Then a gust carried \
the faint smell of pine mulch and chimney smoke — her yard — and she \
bolted through the downpour without hesitating. Twenty minutes later she \
was scratching at the front door, soaked but wagging, as the porch light \
flickered on.
###

### Prompt: {question}
### Keywords:
"""

RESPONSE_CONTINUATION = "\n### Response:\n"


def _truncate_at_separator(text: str) -> str:
    """Return text before the first ``###`` separator, stripped."""
    idx = text.find(SEPARATOR)
    if idx != -1:
        return text[:idx].strip()
    return text.strip()


def generate(
    model,
    tokenizer,
    prompts: list[str],
    n_samples: int,
    *,
    model_name: str,
    temperature_intent: float,
    temperature_response: float,
    max_new_tokens: int = 512,
    intent_max_new_tokens: int = 48,
    few_shot_prompt: str = DEFAULT_FEW_SHOT_PROMPT,
    batch_size: int = 128,
    use_cache: bool = True,
    results_dir: str = "results",
) -> tuple[pl.DataFrame, dict]:
    """Two-stage generation following Ahmed et al., 2025.

    Stage 1 samples comma-separated keyword intents at *temperature_intent*.
    Stage 2 continues from the same assistant context at
    *temperature_response*, so the model conditions on both the original
    prompt and the sampled intent without a separate prompt reformulation.

    Returns (results_df, metadata).
    """
    # assert temperature_intent >= temperature_response, (
    #     f"IFG expects temperature_intent >= temperature_response, "
    #     f"got {temperature_intent} < {temperature_response}"
    # )

    metadata = {
        "id": f"ifg_ti={temperature_intent}_tr={temperature_response}",
        "name": "ifg",
        "model_name": model_name,
        "n_samples": n_samples,
        "temperature_intent": temperature_intent,
        "temperature_response": temperature_response,
        "max_new_tokens": max_new_tokens,
        "intent_max_new_tokens": intent_max_new_tokens,
        "few_shot_prompt": few_shot_prompt,
    }

    if use_cache:
        cached = check_cache(metadata, results_dir)
        if cached is not None:
            print(
                f"  [cached] ifg  t_intent={temperature_intent}  "
                f"t_resp={temperature_response}"
            )
            return cached, metadata

    print(
        f"  Generating ifg  t_intent={temperature_intent}  "
        f"t_resp={temperature_response} ...",
        end=" ",
        flush=True,
    )

    Q = len(prompts)

    # ------------------------------------------------------------------
    # Stage 1: sample keyword intents at high temperature
    # ------------------------------------------------------------------
    intent_user_messages = [
        few_shot_prompt.format(question=q) for q in prompts for _ in range(n_samples)
    ]
    flat_intents_raw = generate_batched(
        intent_user_messages,
        model,
        tokenizer,
        batch_size,
        max_new_tokens=intent_max_new_tokens,
        do_sample=True,
        temperature=temperature_intent,
    )
    flat_intents = [_truncate_at_separator(r) for r in flat_intents_raw]

    # ------------------------------------------------------------------
    # Stage 2: continue from the same context at lower temperature
    # ------------------------------------------------------------------
    # Re-apply the chat template to reconstruct the token prefix, then
    # append the generated keywords + "### Response:" so the model
    # continues the same assistant turn rather than seeing a fresh prompt.
    continuation_texts = []
    for i, user_msg in enumerate(intent_user_messages):
        chat_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        continuation_texts.append(chat_text + flat_intents[i] + RESPONSE_CONTINUATION)

    flat_responses_raw = generate_continuation_batched(
        continuation_texts,
        model,
        tokenizer,
        batch_size,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature_response,
    )
    flat_responses = [_truncate_at_separator(r) for r in flat_responses_raw]

    responses = [flat_responses[q * n_samples : (q + 1) * n_samples] for q in range(Q)]

    print("done")
    return build_tidy_results(prompts, responses), metadata
