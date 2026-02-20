"""Steering-vector method: produce N diverse samples by applying N orthogonal
random perturbations to a chosen residual-stream layer during prompt encoding.

Uses nnsight-style activation patching via a forward hook on the target layer
to steer only the prompt tokens.  Subsequent auto-regressive generation
proceeds unmodified — the steered representations propagate through KV-cache.
"""

from __future__ import annotations

import polars as pl
import torch

from utils import build_tidy_results, check_cache, tokenize_prompts


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
    temperature: float,
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
        "name": "steering",
        "model": model_name,
        "n_samples": n_samples,
        "temperature": temperature,
        "layer": layer,
        "coefficient": coefficient,
        "max_new_tokens": max_new_tokens,
    }

    if use_cache:
        cached = check_cache(metadata, results_dir)
        if cached is not None:
            print(
                f"  [cached] steering  layer={layer} "
                f"coeff={coefficient} temp={temperature}"
            )
            return cached, metadata

    print(
        f"  Generating steering  layer={layer} coeff={coefficient} "
        f"temp={temperature} ...",
        end=" ",
        flush=True,
    )

    d_model = model.config.hidden_size
    vecs = _orthogonal_vectors(n_samples, d_model, device=model.device)
    vecs = (vecs * coefficient).to(model.dtype)  # (n_samples, d_model)

    Q = len(prompts)
    responses: list[list[str]] = [[] for _ in range(Q)]
    target_layer = model.model.layers[layer]

    for sv_idx in range(n_samples):
        sv = vecs[sv_idx]  # (d_model,)
        is_prompt_pass = [True]

        def _hook(_mod, _inp, output, _sv=sv, _flag=is_prompt_pass):
            """Add steering vector only on the first (prompt-encoding) pass."""
            if _flag[0]:
                _flag[0] = False
                return (output[0] + _sv,) + output[1:]

        handle = target_layer.register_forward_hook(_hook)
        try:
            for b in range(0, Q, batch_size):
                batch = prompts[b : b + batch_size]
                inputs = tokenize_prompts(batch, tokenizer, model.device)
                prompt_len = inputs["input_ids"].shape[1]

                is_prompt_pass[0] = True
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                )
                gen = out_ids[:, prompt_len:]
                for j in range(gen.shape[0]):
                    responses[b + j].append(
                        tokenizer.decode(gen[j], skip_special_tokens=True).strip()
                    )
        finally:
            handle.remove()

    print("done")
    return build_tidy_results(prompts, responses), metadata
