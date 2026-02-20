"""Steering-vector method: produce N diverse samples by applying N orthogonal
random perturbations to a chosen residual-stream layer during prompt encoding.

Uses nnsight to trace through the model and add a steering vector to the
target layer's residual stream during the prompt-encoding pass only.
Subsequent auto-regressive token generation proceeds unmodified — the steered
representations propagate through KV-cache.
"""

from __future__ import annotations

import polars as pl
import torch
from nnsight import LanguageModel

from utils import build_tidy_results, check_cache


def _orthogonal_vectors(n: int, d: int, device: torch.device) -> torch.Tensor:
    """Return *n* pairwise-orthonormal row-vectors in R^d via QR decomposition.

    Shape: (n, d).  Requires n <= d.
    """
    assert n <= d, f"Cannot create {n} orthogonal vectors in R^{d}"
    Q, _ = torch.linalg.qr(torch.randn(d, n, device=device))
    vecs = Q[:, :n].T  # (n, d)
    gram = vecs @ vecs.T
    err = (gram - torch.eye(n, device=device)).abs().max().item()
    assert err < 1e-4, f"Orthogonality check failed (max off-diag = {err:.2e})"
    return vecs


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompts: list[str],
    n_samples: int,
    *,
    model_name: str,
    temperature_response: float,
    layer: int,
    coefficient: float = 1.0,
    max_new_tokens: int = 512,
    batch_size: int = 128,
    use_cache: bool = True,
    results_dir: str = "results",
) -> tuple[pl.DataFrame, dict]:
    """Generate N responses per prompt via orthogonal steering vectors.

    Creates *n_samples* orthogonal random unit vectors, scales each by
    *coefficient*, and adds the result to the residual stream at the
    specified *layer* during the prompt-encoding forward pass only.
    Token generation proceeds without further intervention — the steered
    prompt representations are carried forward via KV-cache.  One
    response is sampled per steering direction, yielding *n_samples*
    diverse outputs per prompt.

    Returns (results_df, metadata).
    """
    metadata = {
        "id": f"steering_l={layer}_c={coefficient}_t={temperature_response}",
        "name": "steering",
        "model_name": model_name,
        "n_samples": n_samples,
        "temperature_response": temperature_response,
        "layer": layer,
        "coefficient": coefficient,
        "max_new_tokens": max_new_tokens,
    }

    if use_cache:
        cached = check_cache(metadata, results_dir)
        if cached is not None:
            print(
                f"  [cached] steering  layer={layer} "
                f"coeff={coefficient} t={temperature_response}"
            )
            return cached, metadata

    print(
        f"  Generating steering  layer={layer} coeff={coefficient} "
        f"t={temperature_response} ...",
        end=" ",
        flush=True,
    )

    d_model = model.config.hidden_size
    vecs = _orthogonal_vectors(n_samples, d_model, device=model.device)
    vecs = (vecs * coefficient).to(model.dtype)  # (n_samples, d_model)

    nn_model = LanguageModel(model, tokenizer=tokenizer)

    Q = len(prompts)
    responses: list[list[str]] = [[] for _ in range(Q)]

    chat_texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for p in prompts
    ]
    prompt_token_lens = [len(tokenizer.encode(ct)) for ct in chat_texts]

    for sv_idx in range(n_samples):
        sv = vecs[sv_idx]  # (d_model,)

        for b_start in range(0, Q, batch_size):
            b_texts = chat_texts[b_start : b_start + batch_size]
            b_lens = prompt_token_lens[b_start : b_start + batch_size]
            B = len(b_texts)

            saved = []
            with nn_model.generate(
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature_response,
            ) as tracer:
                for text in b_texts:
                    with tracer.invoke(text):
                        nn_model.model.layers[layer].output[:] += sv
                        saved.append(nn_model.generator.output.save())

            for j in range(B):
                toks = saved[j].squeeze()
                gen_toks = toks[b_lens[j] :]
                resp = tokenizer.decode(gen_toks, skip_special_tokens=True).strip()
                responses[b_start + j].append(resp)

    print("done")
    return build_tidy_results(prompts, responses), metadata
