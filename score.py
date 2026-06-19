import os
import re
import math
import torch
import spacy
import numpy as np
import pandas as pd
from spacy.tokens import Doc
import torch.nn.functional as F
from collections import defaultdict, deque
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
 
 
# ---------------- config ----------------
TOKENIZER_PATH = "/home/shiva/PROJECTS/infill-modeling/tokenizers/word_wikitext-candor.json"
MODEL_PATH     = "/home/shiva/PROJECTS/infill-modeling/models/gpt2-fim-wiki-candor/checkpoint-9274"
 
DATA_PATH  = "SWBD_durationData_scored.csv"      # the per-word table in the screenshot
OUT_PATH   = "SWBD_durationData_scored.csv"
WORD_COL   = "word"                  # lowercased word, matches the tokenizer
GROUP_KEYS = ["swbdID", "uttrID"]    # one utterance = one turn
ORDER_COL  = "word_position"
 
# hyperparameters
CONFIG   = "infill"        # options: forward, backward, or infill 
NOISE    = False            # whether inject distance-scaled noise into the future (suffix) embeddings
DISTANCE = "linear"     # type of distance: linear or dependency       
BETA1    = 1.0              # noise slope       
BETA0    = 1.0              # noise intercept  
RHO      = 0.5              # noise-to-signal budget
SEED     = 0
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------------------
 
tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH).to(DEVICE).eval()
 
# initialize spacy
_nlp = spacy.load("en_core_web_sm")
 
SUF_ID = tokenizer.convert_tokens_to_ids("<suf>")
MID_ID = tokenizer.convert_tokens_to_ids("<mid>")
 
 
# functions
def _parse(words):
    doc = Doc(_nlp.vocab, words=words)
    for _, pipe in _nlp.pipeline:
        doc = pipe(doc)
    return doc
 
 
def _tree_distance(tok_a, tok_b):
    """Compute distance between tokens a and b"""
    depth = {}
    t, d = tok_a, 0
    while True:
        depth[t.i] = d
        # if root
        if t.head.i == t.i:
            break
        t, d = t.head, d + 1
    t, d = tok_b, 0
    while True:
        if t.i in depth:
            return depth[t.i] + d
        if t.head.i == t.i:
            return None
        t, d = t.head, d + 1
 
 
def dependency_distances(prefix, target, suffix):
    """
    Compute dependency distances
    """
    words = (f"{prefix} {target} {suffix}").split()
    target_idx = len(prefix.split())              # target sits right after the prefix
    suffix_start = target_idx + 1
    doc = _parse(words)
    dists = []
    for j in range(suffix_start, len(words)):
        d = _tree_distance(doc[target_idx], doc[j])
        dists.append(d if d is not None else abs(j - target_idx))   # fallback: linear
    return dists
 
 
def noise_fn(E, d, beta1, beta0, rho, rng):
    """
    E: N x D embedding matrix (N = number of words, D = embedding dimension )
    d: vector of distances
    beta1: scaling / slope of noise
    beta0: intercept
    rho: energy of noise / total energy
    """
    N, D = E.shape
    # compute weights
    w = beta1 * d + beta0
    # compute proportions (normalized)
    p = w / w.sum()
    # compute total energy (square of Frobenius norm)
    total_energy = float((E ** 2).sum())
    # compute energy budget, apportioned per dimension
    energy_budget_by_dim = (rho * total_energy) / E.shape[1]
    # per-word sd as a proportion of energy budget
    sigma = np.sqrt(energy_budget_by_dim * p)
    # sample noise vector from Gaussian distribution with mean = 0 and variance = sigma^2
    noise = rng.normal(size=(N, D)) * sigma[:, None]
    return E + noise
 
 
def build_context(prefix, suffix, mode):
    """The FIM context up to and including <mid>; the target gets appended in score_word."""
    if mode == "forward":
        return f"<pre> {prefix} <mid>"
    if mode == "backward":
        return f"<suf> {suffix} <mid>"
    if mode == "infill":
        return f"<pre> {prefix} <suf> {suffix} <mid>"
    raise ValueError(mode)
 
 
def distance_vector(prefix, target, suffix, kind):
    """Per-suffix-word distances from the target (parsed over the full utterance)."""
    n = len(suffix.split())
    if kind == "linear":
        return np.arange(1, n + 1, dtype=float)              # surface position
    if kind == "dependency":
        return np.array(dependency_distances(prefix, target, suffix), dtype=float)
    raise ValueError(kind)
 
 
@torch.no_grad()
def score_word(prefix, target, suffix, mode, noise, distance, beta1, beta0, rho, rng):
    """log p(target | context) under `mode`, optionally noising the future embeddings."""
    context = build_context(prefix, suffix, mode)
    ids = tokenizer(f"{context} {target}", return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
    seq = ids[0].tolist()
    mid_pos = len(seq) - 1 - seq[::-1].index(MID_ID)         # last <mid>
 
    inject = noise and (SUF_ID in seq)
    if inject:
        emb = model.get_input_embeddings()(ids)             # (1, L, D)
        suf_pos = seq.index(SUF_ID)
        suffix_idx = list(range(suf_pos + 1, mid_pos))       # suffix-word token positions
        if suffix_idx:                                       # (empty if target is the last word)
            E = emb[0, suffix_idx].detach().cpu().numpy().astype(float)
            d = distance_vector(prefix, target, suffix, distance)
            assert len(suffix_idx) == len(d), "suffix tokens != suffix words (tokenization)"
            noised = noise_fn(E, d, beta1, beta0, rho, rng)
            emb[0, suffix_idx] = torch.tensor(noised, dtype=emb.dtype, device=emb.device)
        logits = model(inputs_embeds=emb).logits[0]
    else:
        logits = model(ids).logits[0]
 
    logprobs = F.log_softmax(logits, dim=-1)
    total = 0.0
    for t in range(mid_pos + 1, ids.shape[1]):               # target token(s) after <mid>
        total += logprobs[t - 1, seq[t]].item()              # predicted from previous position
    return total
 
 
def score_split(df):
    rng = np.random.default_rng(SEED)
    out = [None] * len(df)
    for _, turn in df.groupby(GROUP_KEYS, sort=False):
        turn = turn.sort_values(ORDER_COL)
        words = turn[WORD_COL].astype(str).tolist()
        for i, row in enumerate(turn.index):
            prefix = " ".join(words[:i])
            suffix = " ".join(words[i + 1:])
            out[df.index.get_loc(row)] = score_word(
                prefix, words[i], suffix, CONFIG, NOISE, DISTANCE, BETA1, BETA0, RHO, rng)
    return out
 
 
if __name__ == "__main__":
    
    cfg = {"forward": "fwd", "backward": "bwd", "infill": "infill"}[CONFIG]
    if NOISE:
        dst = {"linear": "lin", "dependency": "dep"}[DISTANCE]
        col = f"logProb_{cfg}_{dst}_{BETA1}_{BETA0}_{RHO}"   # e.g. logProb_infill_dep_1.0_1.0_0.5
    else:
        col = f"logProb_{cfg}"             

    df = pd.read_csv(DATA_PATH)
    df[col] = score_split(df)
    df.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH}  ({len(df)} words)  col={col}")
    print(df[[WORD_COL, col]].head(12).to_string(index=False))