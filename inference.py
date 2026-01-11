import math
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

TOKENIZER_PATH = "/home/shiva/PROJECTS/word-infill-model-training/tokenizer/candor_tokenizer.json"
MODEL_PATH = "/home/shiva/PROJECTS/word-infill-model-training/models/gpt2-fw-candor/checkpoint-662895"

def score(logits, token):
    """
    Compute log p(w | C) for an AR LM 
    """
    # tokenize w
    token_id = tokenizer.encode(token)
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs[token_id].item()


def prepare_infill_inputs(context,target_token,target_position,use_prefix=True,use_suffix=True):
    """
    Generate infill-augmented input for inference
    """
    context_tokens = context.split()
    assert context_tokens[target_position] == target_token, "mistmatch between target token and position"
    
    if use_prefix:
        prefix = "<PRE>" + " " + " ".join(context_tokens[:target_position]) + " "
    else:
        prefix = ""
    
    if use_suffix:
        suffix = "<SUF>" + " " + " ".join(context_tokens[target_position + 1:]) + " "
    else:
        suffix = ""

    modified_context = prefix + suffix  + "<MID>"
    
    return modified_context, target_token


def get_embeddings(model, tokenizer, context):
    """
    Get input embeddings of context from model
    """
    input_ids = torch.tensor(tokenizer.encode(context)).unsqueeze(0)
    embeddings = model.get_input_embeddings()(input_ids)
    return embeddings

def cosine_sim_per_token(embeds, noisy_embeds):
    """
    Calculates cosine similarity for each token position.
    """
    # remove batch dim
    e = embeds[0]
    n = noisy_embeds[0]
    return F.cosine_similarity(e, n, dim=-1)

def inject_noise_in_prefix(embeds,start_idx,end_idx,A=1,lam=0.5):
    """
    Exponential decay in prefix.
    """
    noisy_embeds = embeds.clone()
    _, seq_len, D = embeds.shape

    for i in range(start_idx, end_idx):
        # recency of token
        distance = (end_idx - 1) - i
        # A * exp{-lambda * distance}
        sigma = A * (1 - math.exp(-lam * distance))
        # generate noise and decay noise 
        noisy_embeds[0, i,:] += torch.randn(D, device=embeds.device) * sigma

    return noisy_embeds

def prepare_noisy_prefix(model,tokenizer,context,infill=False):
    context_tokens = context.split()
    if infill:
        start_idx = 3
        end_idx = context_tokens.index("<MID>")
    else:
        start_idx = 2
        end_idx = len(context_tokens)
    
    embeds = get_embeddings(model,tokenizer,context)    
    noisy_embeds = inject_noise_in_prefix(embeds,start_idx,end_idx)
    return embeds, noisy_embeds



tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)

#context = "<eos> B: <PRE> The capital of France is <MID>"
#embeds,noisy_embeds = prepare_noisy_prefix(model,tokenizer,context,infill=True)
#sims = cosine_sim_per_token(embeds, noisy_embeds)

#for i, sim in enumerate(sims):#
#    print(i, sim.item())

# 
data = pd.read_csv('SWBD_DurAnalysisData.csv')
data.head()



