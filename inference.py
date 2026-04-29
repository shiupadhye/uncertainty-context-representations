import os
import re
import math
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from collections import defaultdict, deque
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast


TOKENIZER_PATH = "/home/shiva/PROJECTS/word-infill-model-training/tokenizer/candor_tokenizer.json"
MODEL_PATH = "/home/shiva/PROJECTS/word-infill-model-training/models/gpt2-fw-candor/checkpoint-662895"

tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)

# CONFIG
noise  = True

# exp scaling noise params
A  = 0      
lam  = 0

# linear scaling noise
lam0  = 0         # linear intercept
beta  = 0.1    # linear slope

# context window and LM parameters
lm_mode  = 'causal'    # causal, bidirectional
context_type = 'prefix'  #prefix, suffix
noise_type   = 'linear'   # linear, dependency, random

print(f"LM Mode: {lm_mode}\nContext type: {context_type}\nNoise type: {noise_type}")
print(f"A = {A:.2f}  lambda = {lam:.2f}  beta = {beta}  lam0 = {lam0}")

data = pd.read_csv('SWBD_durationData_depparsed_subset2K_linNoise_scores.csv')

norms_by_distance = defaultdict(list)


# functions
def score(logits, target):
    target_id = tokenizer.encode(target)[0]
    log_probs = F.log_softmax(logits[0, -1], dim=-1)
    return log_probs[target_id].item()

def cosine_sim_per_token(embeds, noisy_embeds):
    return F.cosine_similarity(embeds[0], noisy_embeds[0], dim=-1)

def get_embeddings(model, tokenizer, context):
    input_ids = torch.tensor(tokenizer.encode(context)).unsqueeze(0)
    return model.get_input_embeddings()(input_ids)

def extract_indices(input_str):
    tokens = input_str.split()
    pre_idx = tokens.index("<PRE>") if "<PRE>" in tokens else None
    suf_idx = tokens.index("<SUF>") if "<SUF>" in tokens else None
    mid_idx = tokens.index("<MID>") if "<MID>" in tokens else None
    return pre_idx, suf_idx, mid_idx

def build_dep_graph(uttr_df):
    graph = defaultdict(list)
    for _, row in uttr_df.iterrows():
        i, h = int(row["token_id"]), int(row["head_token_id"])
        if h != -1:
            graph[i].append(h)
            graph[h].append(i)
    return graph

def compute_dep_distances(graph, current_token_id):
    dist = {current_token_id: 0}
    queue = deque([current_token_id])
    while queue:
        u = queue.popleft()
        for v in graph[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist

def normalize(token):
    token = token.lower().strip()
    return re.sub(r"^[^\w]+|[^\w]+$", "", token)

def lm_idx_to_dep_id(lm_tokens, dep_df):
    special = {"<eos>", "<PRE>", "<SUF>", "<MID>", "A:", "B:"}
    dep_words = dep_df["word"].tolist()
    dep_ids   = dep_df["token_id"].tolist()
    mapping, dep_ptr = {}, 0
    for i, tok in enumerate(lm_tokens):
        if tok in special:
            mapping[i] = None
            continue
        norm_tok = normalize(tok)
        while dep_ptr < len(dep_words) and normalize(dep_words[dep_ptr]) != norm_tok:
            dep_ptr += 1
        if dep_ptr == len(dep_words):
            mapping[i] = None
        else:
            mapping[i] = dep_ids[dep_ptr]
            dep_ptr += 1
    return mapping

def L2_norm(orig_embeds, noisy_embeds):
    return torch.norm((noisy_embeds - orig_embeds)[0], dim=-1)

def display_norm(norms, context_tokens):
    for i, tok in enumerate(context_tokens):
        val = norms[i].item() if hasattr(norms[i], "item") else float(norms[i])
        print(f"{tok}\t{val:.6f}")

def record_L2_by_distance(norms, context_tokens, context_type, noise_type,
                          pre_idx, suf_idx, mid_idx, lm_to_dep=None, dep_dist=None):
    special = {"<eos>", "<PRE>", "<SUF>", "<MID>", "A:", "B:"}

    if noise_type in ("linear", "random"):
        start_idx, end_idx = (pre_idx, mid_idx) if context_type == "prefix" else (suf_idx, mid_idx)

    for i_tok, tok in enumerate(context_tokens):
        if tok in special:
            continue
        val = norms[i_tok].item()

        if noise_type in ("linear", "random"):
            if not (start_idx < i_tok < end_idx):
                continue
            if context_type == "prefix":
                d = (end_idx - 1) - i_tok + 1
            else:
                d = i_tok - (start_idx + 1) + 1
        else:
            dep_id = lm_to_dep.get(i_tok, None)
            if dep_id is None:
                continue
            d = dep_dist.get(dep_id, None)

        if d is not None:
            norms_by_distance[d].append(val)


# ---------- noise functions ----------
## linear scaling functions
def prefix_lin_noise(embeds, start_idx, end_idx, beta, lam0):
    noisy = embeds.clone()
    _, _, D = embeds.shape
    for i in range(start_idx + 1, end_idx):
        d = (end_idx - 1) - i + 1
        sigma = beta * d + lam0
        noisy[0, i, :] += torch.randn(D, device=embeds.device) * sigma
    return noisy

def suffix_lin_noise(embeds, start_idx, end_idx, beta, lam0):
    noisy = embeds.clone()
    _, _, D = embeds.shape
    for i in range(start_idx + 1, end_idx):
        d = i - (start_idx + 1) + 1
        sigma = beta * d + lam0
        noisy[0, i, :] += torch.randn(D, device=embeds.device) * sigma
    return noisy

def prefix_lin_noise_dep(embeds, start_idx, end_idx, beta, lam0, lm_to_dep, dep_dist):
    noisy = embeds.clone()
    _, _, D = embeds.shape
    for i in range(start_idx, end_idx):
        dep_id = lm_to_dep.get(i)
        if dep_id is None:
            continue
        d = dep_dist.get(dep_id)
        if d is None:
            continue
        sigma = beta * d + lam0
        if sigma > 0:
            noisy[0, i, :] += torch.randn(D, device=embeds.device) * sigma
    return noisy

def suffix_lin_noise_dep(embeds, start_idx, end_idx, beta, lam0, lm_to_dep, dep_dist):
    noisy = embeds.clone()
    _, _, D = embeds.shape
    for i in range(start_idx + 1, end_idx):
        dep_id = lm_to_dep.get(i)
        if dep_id is None:
            continue
        d = dep_dist.get(dep_id)
        if d is None:
            continue
        sigma = beta * d + lam0
        if sigma > 0:
            noisy[0, i, :] += torch.randn(D, device=embeds.device) * sigma
    return noisy

## random baseline
def _uniform_noise(embeds, start_idx, end_idx, beta, lam0):
    noisy = embeds.clone()
    _, _, D = embeds.shape
    N = end_idx - start_idx - 1
    sigma = beta * (N + 1) / 2 + lam0
    if sigma > 0:
        for i in range(start_idx + 1, end_idx):
            noisy[0, i, :] += torch.randn(D, device=embeds.device) * sigma
    return noisy

# for prefix and suffix versions
prefix_rand_noise = _uniform_noise
suffix_rand_noise = _uniform_noise

## exponential scaling functions
def prefix_exp_noise(embeds, start_idx, end_idx, A, lam):
    noisy = embeds.clone()
    _, _, D = embeds.shape
    for i in range(start_idx + 1, end_idx):
        d = (end_idx - 1) - i + 1
        sigma = A * (1 - math.exp(-lam * d))
        noisy[0, i, :] += torch.randn(D, device=embeds.device) * sigma
    return noisy

def suffix_exp_noise(embeds, start_idx, end_idx, A, lam):
    noisy = embeds.clone()
    _, _, D = embeds.shape
    for i in range(start_idx + 1, end_idx):
        d = i - (start_idx + 1) + 1
        sigma = A * (1 - math.exp(-lam * d))
        noisy[0, i, :] += torch.randn(D, device=embeds.device) * sigma
    return noisy


###
NOISE_FNS = {
    ("prefix", "linear"):     lambda e, s, t, ltd, dd: prefix_lin_noise(e, s, t, beta, lam0),
    ("prefix", "dependency"): lambda e, s, t, ltd, dd: prefix_lin_noise_dep(e, s, t, beta, lam0, ltd, dd),
    ("prefix", "random"):     lambda e, s, t, ltd, dd: prefix_rand_noise(e, s, t, beta, lam0),
    ("suffix", "linear"):     lambda e, s, t, ltd, dd: suffix_lin_noise(e, s, t, beta, lam0),
    ("suffix", "dependency"): lambda e, s, t, ltd, dd: suffix_lin_noise_dep(e, s, t, beta, lam0, ltd, dd),
    ("suffix", "random"):     lambda e, s, t, ltd, dd: suffix_rand_noise(e, s, t, beta, lam0),
}

def noise_region(lm_mode, context_type, pre_idx, suf_idx, mid_idx):
    if context_type == "prefix":
        return pre_idx, (suf_idx if lm_mode == "bidirectional" else mid_idx)
    return suf_idx, mid_idx



def iter_contexts(uttrWords, role, lm_mode, context_type):
    """Yields (context, target, word_idx_in_uttr) for each iteration."""
    if lm_mode == "causal" and context_type == "prefix":
        prefix = f"<eos> {role}: <PRE> "
        for i in range(-1, len(uttrWords) - 1):
            if i >= 0:
                prefix += uttrWords[i] + " "
            yield prefix + "<MID>", uttrWords[i + 1], i + 1

    elif lm_mode == "causal" and context_type == "suffix":
        for i in range(len(uttrWords)):
            context = "<SUF> " + " ".join(uttrWords[i + 1:]) + " <MID>"
            yield context, uttrWords[i], i

    elif lm_mode == "bidirectional":
        prefix = f"<eos> {role}: <PRE> "
        for i in range(-1, len(uttrWords) - 1):
            if i >= 0:
                prefix += uttrWords[i] + " "
            suffix = "<SUF> " + " ".join(uttrWords[i + 2:]) + " <MID>"
            sep = " " if i == -1 else ""
            yield prefix + sep + suffix, uttrWords[i + 1], i + 1

    else:
        raise ValueError(f"unknown combo: lm_mode={lm_mode}, context_type={context_type}")


def score_context(context, target, uttr_df, uttr_graph, word_idx):
    pre_idx, suf_idx, mid_idx = extract_indices(context)
    context_tokens = context.split()
    orig_embeds = get_embeddings(model, tokenizer, context)

    print(context)
    print(target)

    if not noise:
        final_embeds = orig_embeds
    else:
        start_idx, end_idx = noise_region(lm_mode, context_type, pre_idx, suf_idx, mid_idx)

        lm_to_dep = dep_dist = None
        if noise_type == "dependency":
            current_token_id = int(uttr_df.iloc[word_idx]["token_id"])
            dep_dist  = compute_dep_distances(uttr_graph, current_token_id)
            lm_to_dep = lm_idx_to_dep_id(context_tokens, uttr_df)

        noisy_embeds = NOISE_FNS[(context_type, noise_type)](
            orig_embeds, start_idx, end_idx, lm_to_dep, dep_dist)

        norms = L2_norm(orig_embeds, noisy_embeds)
        display_norm(norms, context_tokens)
        record_L2_by_distance(norms, context_tokens, context_type, noise_type,
                              pre_idx, suf_idx, mid_idx, lm_to_dep, dep_dist)
        final_embeds = noisy_embeds

    with torch.no_grad():
        outputs = model(inputs_embeds=final_embeds)
        return score(outputs.logits, target)


uttrIDs = np.unique(data['uttrID'].values)

contexts = []
targets = []
scores_list = []

for uttrID in uttrIDs:
    if uttrID % 100 == 0:
        print(uttrID)
    uttr_df   = data[data['uttrID'] == uttrID]
    role      = uttr_df['role'].values[0].split("_")[-1]
    uttrWords = list(uttr_df['word'].values)
    uttr_graph = build_dep_graph(uttr_df) if noise_type == 'dependency' else None

    for context, target, word_idx in iter_contexts(uttrWords, role, lm_mode, context_type):
        target_score = score_context(context, target, uttr_df, uttr_graph, word_idx)
        contexts.append(context)
        targets.append(target)
        scores_list.append(target_score)


# save
if noise:
    rows = [{"distance": int(d), "mean_L2": float(np.mean(v)), "N": len(v)}
            for d, v in norms_by_distance.items()]
    df_stats = pd.DataFrame(rows).sort_values("distance")
    out_path = f"mean_L2_by_distance_{noise_type}_{context_type}_{lm_mode}_B{beta}_lam0{lam0}.csv"
    df_stats.to_csv(os.path.join("norms", out_path), index=False)


# output scores
#col_prefix = "past_noNoise"
col_prefix = f"past_surf_beta{beta}"
subset = data[data['uttrID'].isin(uttrIDs)]
subset[f'{col_prefix}_context'] = contexts
subset[f'{col_prefix}_target']  = targets
subset[f'{col_prefix}_score']   = scores_list
subset.to_csv('SWBD_durationData_depparsed_subset2K_linNoise_scores.csv', index=False)