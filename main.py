# %%
import torch
from dotenv import load_dotenv
from einops import rearrange
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()
# %%
model_name = "google/gemma-3-1b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name  # , dtype=torch.bfloat16, device_map="cpu"
)
model.config.pad_token_id = tokenizer.pad_token_id

# %%
QUESTIONS = [
    "Who was the first president of the United States?",
    "What can you tell me about the history of the Earth?",
    "What is the speed of light in a vacuum?",
    "Who wrote the play Romeo and Juliet?",
    "What is the largest planet in our solar system?",
]


# %%
def generate_responses(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    questions: list[str],
    n: int,
    **kwargs,
) -> torch.Tensor:
    """Generate n responses for each question in a single batched forward pass.

    Builds a flat batch of Q*N prompts (each question repeated n times),
    generates all responses simultaneously, then reshapes to (Q, N, S).

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        questions: List of Q prompt strings.
        n: Number of responses per question.
        **kwargs: Forwarded to model.generate() (e.g. temperature, do_sample).

    Returns:
        tokens_QNS: Long tensor of shape (Q, N, S) with generated token ids,
                     where S is the max generated sequence length (right-padded).
    """
    Q = len(questions)
    prompts = [q for q in questions for _ in range(n)]

    prev_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    tokenizer.padding_side = prev_padding_side

    with torch.no_grad():
        output_ids = model.generate(**inputs, **kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    generated_BatS = output_ids[:, prompt_len:]  # (Q*N, S)

    tokens_QNS = rearrange(generated_BatS, "(q n) s -> q n s", q=Q, n=n)
    return tokens_QNS


# %%
def compute_entropies(
    model: AutoModelForCausalLM,
    tokens_QNS: torch.Tensor,
) -> torch.Tensor:
    """Compute mean per-token entropy for each generated response.

    Feeds the generated tokens through the model and measures the entropy
    of the predicted next-token distribution at every position, averaged
    over non-padding positions.

    Args:
        model: The language model.
        tokens_QNS: Long tensor of shape (Q, N, S) with generated token ids.

    Returns:
        entropies_QN: Float tensor of shape (Q, N) with mean per-token entropy.
    """
    Q, N, S = tokens_QNS.shape
    tokens_BS = rearrange(tokens_QNS, "q n s -> (q n) s")

    pad_id = model.config.pad_token_id
    if pad_id is not None:
        mask_BS = (tokens_BS != pad_id).long()
    else:
        mask_BS = torch.ones_like(tokens_BS)

    with torch.no_grad():
        logits_BSV = model(
            input_ids=tokens_BS.to(model.device),
            attention_mask=mask_BS.to(model.device),
        ).logits.float()

    log_probs_BSV = torch.log_softmax(logits_BSV, dim=-1)
    probs_BSV = torch.exp(log_probs_BSV)
    entropy_BS = -(probs_BSV * log_probs_BSV).sum(dim=-1)  # (B, S)

    mask_float_BS = mask_BS.float().to(entropy_BS.device)
    entropies_B = (entropy_BS * mask_float_BS).sum(dim=-1) / mask_float_BS.sum(
        dim=-1
    ).clamp(min=1)

    entropies_QN = rearrange(entropies_B, "(q n) -> q n", q=Q, n=N)
    return entropies_QN


# %%

tokens_QNS = generate_responses(
    model,
    tokenizer,
    QUESTIONS,
    n=3,
    max_new_tokens=128,
    do_sample=True,
    temperature=0.7,
)
print("tokens_QNS shape:", tokens_QNS.shape)

entropies_QN = compute_entropies(model, tokens_QNS)
print("entropies_QN shape:", entropies_QN.shape)
print("entropies_QN:\n", entropies_QN)

# %%
