# %%
import lovely_tensors as lt
import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv
from einops import rearrange
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

lt.monkey_patch()
load_dotenv()

# %%
model_name = "google/gemma-3-4b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name, dtype=torch.bfloat16, device_map="cuda:0"
)
model.config.pad_token_id = tokenizer.pad_token_id

emb_model = SentenceTransformer("all-MiniLM-L6-v2")

nli_model_name = "microsoft/deberta-large-mnli"
nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(
    "cuda:0"
)
nli_model.eval()

# %%
# Prompts modelled on NoveltyBench NB-Curated categories:
#   - Randomness: many equally valid discrete answers
#   - Factual knowledge: underspecified, many correct answers
#   - Creative writing: open-ended generation
#   - Subjectivity: opinion-based
QUESTIONS = [
    # Randomness
    "Pick a random number between 1 and 100.",
    "Name a random color.",
    # Factual knowledge (underspecified → many valid answers)
    "Name a capital city in Europe.",
    "Name a species of bird.",
    "Tell me a historical event from the 20th century.",
    # Creative writing
    "Tell me a very short fictional story in 2-3 sentences.",
    "Give me an original metaphor for loneliness.",
    "Write a riddle.",
    # Subjectivity
    "What is a fun hobby to pick up?",
    "Recommend me an interesting place to travel to.",
]

N = 8  # responses per question — matches NoveltyBench's 8 human baselines


# %%
def generate_responses(questions, n, **kwargs):
    prompts = [q for q in questions for _ in range(n)]

    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, **kwargs)

    generated = output_ids[:, inputs["input_ids"].shape[1] :]
    return rearrange(generated, "(q n) s -> q n s", q=len(questions), n=n)


def decode_responses(tokens_QNS):
    Q, N, _ = tokens_QNS.shape
    return [
        [
            tokenizer.decode(tokens_QNS[q, n], skip_special_tokens=True).strip()
            for n in range(N)
        ]
        for q in range(Q)
    ]


# %%
def compute_embedding_diversity(tokens_QNS):
    """Mean pairwise cosine distance among N responses, per question. Returns (Q,)."""
    Q, N, _ = tokens_QNS.shape
    texts = [t for q_texts in decode_responses(tokens_QNS) for t in q_texts]

    embs = emb_model.encode(texts, convert_to_tensor=True)
    embs = rearrange(embs, "(q n) d -> q n d", q=Q, n=N)
    embs = torch.nn.functional.normalize(embs, p=2, dim=-1)

    # (Q, N, N) pairwise cosine similarities
    sims = torch.bmm(embs, embs.transpose(1, 2))

    # mean over off-diagonal pairs → within-prompt diversity
    mask = ~torch.eye(N, dtype=torch.bool).unsqueeze(0).expand(Q, -1, -1)
    return (1.0 - sims)[mask].reshape(Q, N * (N - 1)).mean(dim=-1)


# %%
def _bidirectional_entailment(text_a: str, text_b: str) -> bool:
    """True if a entails b AND b entails a (DeBERTa-large-mnli label 2 = entailment)."""
    for premise, hypothesis in [(text_a, text_b), (text_b, text_a)]:
        inputs = nli_tokenizer(
            premise, hypothesis, return_tensors="pt", truncation=True, max_length=512
        ).to(nli_model.device)
        with torch.no_grad():
            pred = nli_model(**inputs).logits.argmax(dim=-1).item()
        if pred != 2:
            return False
    return True


def compute_semantic_diversity(tokens_QNS):
    """NLI-based distinct semantic classes among N responses, per question.

    Following Kuhn et al. (Nature 2024) / NoveltyBench:
      1. For each question, greedily cluster N responses via bidirectional entailment.
      2. Return the number of distinct clusters (= "distinct_k") per question.

    This counts *within-prompt* semantic equivalence classes — exactly what
    NoveltyBench measures. Higher = more semantically diverse.

    Returns (Q,) tensor of floats (number of distinct clusters per question).
    """
    responses = decode_responses(tokens_QNS)
    Q = len(responses)
    distinct = []

    for q in range(Q):
        # Greedy clustering: assign each response to first matching cluster
        # or create a new one. O(N * num_clusters) NLI calls per question.
        representatives: list[int] = []  # index of cluster representative
        for i in range(len(responses[q])):
            matched = False
            for rep in representatives:
                if _bidirectional_entailment(responses[q][i], responses[q][rep]):
                    matched = True
                    break
            if not matched:
                representatives.append(i)
        distinct.append(float(len(representatives)))

    return torch.tensor(distinct)


# %%
temperatures = np.linspace(0.01, 1.0, 12)
results = {
    "emb_mean": [],
    "emb_sem": [],
    "nli_mean": [],
    "nli_sem": [],
}

Q = len(QUESTIONS)

for temp in temperatures:
    print(f"temp={temp:.2f} ... ", end="", flush=True)
    tokens = generate_responses(
        QUESTIONS, n=N, max_new_tokens=128, do_sample=True, temperature=float(temp)
    )

    # Embedding diversity: (Q,) → mean ± SEM across questions
    emb_Q = compute_embedding_diversity(tokens)
    results["emb_mean"].append(emb_Q.mean().item())
    results["emb_sem"].append(emb_Q.std().item() / np.sqrt(Q))

    # NLI diversity: (Q,) → mean ± SEM across questions
    nli_Q = compute_semantic_diversity(tokens)
    results["nli_mean"].append(nli_Q.mean().item())
    results["nli_sem"].append(nli_Q.std().item() / np.sqrt(Q))

    print(
        f"emb={results['emb_mean'][-1]:.4f}±{results['emb_sem'][-1]:.4f}  "
        f"distinct={results['nli_mean'][-1]:.2f}±{results['nli_sem'][-1]:.2f}/{N}"
    )

for k in results:
    results[k] = np.array(results[k])

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Left panel: embedding cosine distance
ax1.plot(temperatures, results["emb_mean"], "o-", color="#4C72B0", ms=5, lw=1.5)
ax1.fill_between(
    temperatures,
    results["emb_mean"] - results["emb_sem"],
    results["emb_mean"] + results["emb_sem"],
    alpha=0.2,
    color="#4C72B0",
)
ax1.set_xlabel("Temperature")
ax1.set_ylabel("Mean pairwise cosine distance")
ax1.set_title("Embedding diversity (SBERT)")
ax1.set_xlim(0, 1.05)
ax1.spines[["top", "right"]].set_visible(False)

# Right panel: NLI distinct clusters
ax2.plot(temperatures, results["nli_mean"], "s-", color="#DD8452", ms=5, lw=1.5)
ax2.fill_between(
    temperatures,
    results["nli_mean"] - results["nli_sem"],
    results["nli_mean"] + results["nli_sem"],
    alpha=0.2,
    color="#DD8452",
)
ax2.set_xlabel("Temperature")
ax2.set_ylabel(f"Distinct semantic classes (out of {N})")
ax2.set_title("Semantic diversity (NLI clustering)")
ax2.set_xlim(0, 1.05)
ax2.set_ylim(0, N + 0.5)
ax2.axhline(N, color="gray", ls="--", lw=0.8, label=f"max = {N}")
ax2.legend(frameon=False)
ax2.spines[["top", "right"]].set_visible(False)

fig.suptitle(
    f"{model_name}  ·  N={N} responses/prompt  ·  {Q} prompts  ·  ±1 SEM",
    fontsize=11,
    y=1.02,
)
fig.tight_layout()
fig.savefig("diversity_vs_temperature.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved diversity_vs_temperature.png")

# %%
