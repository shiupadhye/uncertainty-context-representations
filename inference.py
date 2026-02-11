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

def score(logits, target):
    """
    Compute log p(token | context) from AR LM logits.
    """
    target_id = tokenizer.encode(target)[0]
    last_logits = logits[0, -1]          
    log_probs = F.log_softmax(last_logits, dim=-1)
    return log_probs[target_id].item()

def cosine_sim_per_token(embeds, noisy_embeds):
    """
    Calculates cosine similarity for each token position.
    """
    # remove batch dim
    e = embeds[0]
    n = noisy_embeds[0]
    return F.cosine_similarity(e, n, dim=-1)

def get_embeddings(model, tokenizer, context):
    """
    Get input embeddings of context from model
    """
    input_ids = torch.tensor(tokenizer.encode(context)).unsqueeze(0)
    embeddings = model.get_input_embeddings()(input_ids)
    return embeddings

def extract_indices(input_str):
    """
    Returns positions of <PRE>, <SUF>, and <MID> tokens in input string
    """
    tokens = input_str.split()
    pre_idx = tokens.index("<PRE>") if "<PRE>" in tokens else None
    suf_idx = tokens.index("<SUF>") if "<SUF>" in tokens else None
    mid_idx = tokens.index("<MID>") if "<MID>" in tokens else None
    return pre_idx, suf_idx, mid_idx

def build_dep_graph(uttr_df):
    from collections import defaultdict
    graph = defaultdict(list)

    for _, row in uttr_df.iterrows():
        i = int(row["token_id"])
        h = int(row["head_token_id"])
        
        if h != -1:
            graph[i].append(h)
            graph[h].append(i)
    
    return graph

def compute_dep_distances(graph, current_token_id):
    """
    Returns dependency distance to current token
    """
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
    """
    Function for normalizing text
    """
    token = token.lower().strip()
    token = re.sub(r"^[^\w]+|[^\w]+$", "", token)  
    return token

def lm_idx_to_dep_id(lm_tokens, dep_df):
    """
    Align LM tokens to dependency tokens by scanning for matches.
    Assumes dependency tokens appear in order within LM tokens.
    """

    special_tokens={"<eos>", "<PRE>", "<SUF>", "<MID>", "A:","B:"}
    lm_idx_to_dep_id = {}
    
    dep_words = dep_df["word"].tolist()
    dep_ids = dep_df["token_id"].tolist()
    
    dep_ptr = 0
    
    for i, tok in enumerate(lm_tokens):
        if tok in special_tokens:
            lm_idx_to_dep_id[i] = None
            continue
        
        # advance dep_ptr until we find a match
        norm_tok = normalize(tok)
        while dep_ptr < len(dep_words) and normalize(dep_words[dep_ptr]) != norm_tok:        
            dep_ptr += 1
        
        if dep_ptr == len(dep_words):
            # token not found in dep tokens
            lm_idx_to_dep_id[i] = None
        else:
            lm_idx_to_dep_id[i] = dep_ids[dep_ptr]
            dep_ptr += 1
    
    return lm_idx_to_dep_id

def logits_from_embeds(model, embeds, attention_mask=None):
    """
    Run the LM forward pass using *input embeddings* instead of input_ids.
    """
    with torch.no_grad():
        outputs = model(inputs_embeds=embeds)
        logits = outputs.logits
    return logits

def prefix_noise(embeds, start_idx, end_idx, A, lam):
    """
    Injects noise in prefix embeddings
    embeds: model embeddings
    start_idx: position of <PRE> token
    end_idx: position of <SUF> token
    A: scaling constant
    lam: decay rate
    """
    noisy_embeds = embeds.clone()
    _, seq_len, D = embeds.shape

    for i in range(start_idx+1, end_idx):
        # distance from the current word (right before <MID>)
        distance = (end_idx - 1) - i + 1   # 0 for the token right before <MID>
        # noise scale increases with distance
        sigma = A * (1 - math.exp(-lam * distance))
        # add Gaussian noise
        noisy_embeds[0, i, :] += torch.randn(D, device=embeds.device) * sigma

    return noisy_embeds

def suffix_noise(embeds, start_idx, end_idx, A, lam):
    """
    Injects noise in suffix embeddings
    embeds: model embeddings
    start_idx: position of <SUF>
    end_idx: position of <MID>
    A: scaling constant
    lam: decay rate
    """
    noisy_embeds = embeds.clone()
    _, seq_len, D = embeds.shape

    # Suffix region: (start_idx + 1) ... (end_idx - 1)
    for i in range(start_idx + 1, end_idx):
        # distance from the token right after <SUF>
        distance = i - (start_idx + 1) + 1 # 0 for the token right after <SUF>
        # noise scale increases with distance
        sigma = A * (1 - math.exp(-lam * distance))
        # add Gaussian noise
        noisy_embeds[0, i, :] += torch.randn(D, device=embeds.device) * sigma
    return noisy_embeds


def prefix_noise_dep(embeds, start_idx, end_idx, A, lam, lm_idx_to_dep_id, dep_dist):
    """
    Injects noise in prefix embeddings based on dependency distance
    embeds: model embeddings
    start_idx: position of <SUF>
    end_idx: position of <MID>
    A: scaling constant
    lam: decay rate
    lm_idx_to_dep_id: lookup table that maps LM position to dep id
    dep dist: dict containing dependency distances to the current word
    """
    noisy_embeds = embeds.clone()
    _, _, D = embeds.shape

    for i in range(start_idx, end_idx):
        dep_id = lm_idx_to_dep_id.get(i, None)
        if dep_id is None:
            continue
        d = dep_dist.get(dep_id, None)
        if d is None:
            continue

        sigma = A * (1 - math.exp(-lam * d))
        if sigma > 0:
            noisy_embeds[0, i, :] += torch.randn(D, device=embeds.device) * sigma

    return noisy_embeds

def suffix_noise_dep(embeds, start_idx, end_idx, A, lam, lm_idx_to_dep_id, dep_dist):
    """
    Injects noise in suffix embeddings based on dependency distance
    embeds: model embeddings
    start_idx: position of <SUF>
    end_idx: position of <MID>
    A: scaling constant
    lam: decay rate
    lm_idx_to_dep_id: lookup table that maps LM position to dep id
    dep dist: dict containing dependency distances to the current word
    """
    noisy_embeds = embeds.clone()
    _, _, D = embeds.shape

    for i in range(start_idx + 1, end_idx):
        dep_id = lm_idx_to_dep_id.get(i, None)
        if dep_id is None:
            continue
        d = dep_dist.get(dep_id, None)
        if d is None:
            continue

        sigma = A * (1 - math.exp(-lam * d))
        if sigma > 0:
            noisy_embeds[0, i, :] += torch.randn(D, device=embeds.device) * sigma

    return noisy_embeds

def L2_norm(orig_embeds, noisy_embeds):
    """
    Computes L2 norm between original and noisy embeddings
    """
    diff = noisy_embeds - orig_embeds
    return torch.norm(diff[0], dim=-1)


def display_norm(norms,context_tokens):
    """
    Displays L2_norm between orig and noisy embeddings
    """
    for i, tok in enumerate(context_tokens):
        val = norms[i].item() if hasattr(norms[i], "item") else float(norms[i])
        print(f"{tok}\t{val:.6f}")


tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)

# noise scalar: 1.0 (noisy), 0.0 (no noise)
A = 1.0
lam = 0.8
# causal or bidirectional
lm_mode = 'causal'
# prefix or suffix
context_type = 'suffix'
# temporal or dependency
noise_type = 'dependency'

print("LM Mode: %s" % lm_mode)
print("Context type: %s" % context_type)
print("Noise type: %s" % noise_type)
print("A = %.2f" % A)
print("Lambda = %.2f" % lam)


data = pd.read_csv('SWBD_DurData_depparsed100_scores.csv')
uttrIDs = np.unique(data['uttrID'].values)
subset_uttrIDs = uttrIDs

contexts = []
targets = []
scores = []
for uttrID in subset_uttrIDs:
    uttr_df = data[data['uttrID'] == uttrID]
    role = uttr_df['spID'].values[0].split("_")[-1]
    uWordIDs = list(uttr_df['uWordID'].values)
    uttrWords = list(uttr_df['word'].values)

    # string version of utterance
    uttr = " ".join(uttrWords)
    # positions of demarcating tokens
    pre_idx, suf_idx, mid_idx = extract_indices(uttr)
    # orig uttr tokens
    uttr_tokens = uttr.split()
    if noise_type == 'dependency':
        uttr_graph = build_dep_graph(uttr_df)

    if lm_mode == 'causal':
        if context_type == 'prefix':
            prefix = "<eos>" + " " + role + ": " + "<PRE>" + " " 
            i = -1
            while i < len(uttrWords) - 1:
                if i == -1:
                    context = prefix + "<MID>"
                    target = uttrWords[0]

                elif i > -1:
                    prefix += uttrWords[i] + " " 
                    context = prefix + "<MID>"
                    target = uttrWords[i+1]

                # sanity check
                print(context)
                print(target)

                pre_idx, suf_idx, mid_idx = extract_indices(context)
                orig_embeds = get_embeddings(model, tokenizer, context)
                context_tokens = context.split()

                if A == 0:
                    print("No noise injection")
                    with torch.no_grad():
                        outputs = model(inputs_embeds = orig_embeds)
                        logits = outputs.logits
                        target_score = score(logits,target)
                        contexts.append(context)
                        targets.append(target)
                        scores.append(target_score)
                
                else:
                    if noise_type == 'temporal':
                        noisy_embeds = prefix_noise(orig_embeds, start_idx=pre_idx, end_idx=mid_idx, A=A,lam=lam)
                        norms = L2_norm(orig_embeds, noisy_embeds)
                        display_norm(norms,context_tokens)

                        with torch.no_grad():
                            outputs = model(inputs_embeds = noisy_embeds)
                            logits = outputs.logits
                            target_score = score(logits,target)
                            contexts.append(context)
                            targets.append(target)
                            scores.append(target_score)

                    if noise_type == 'dependency':
                        current_token_id = int(uttr_df.iloc[i+1]["token_id"])
                        dep_dist = compute_dep_distances(uttr_graph,current_token_id)
                        lm_to_dep = lm_idx_to_dep_id(context_tokens, uttr_df)

                        noisy_embeds = prefix_noise_dep(orig_embeds,start_idx=pre_idx,end_idx=mid_idx,
                        A=A,lam=lam,lm_idx_to_dep_id=lm_to_dep,dep_dist=dep_dist)

                        norms = L2_norm(orig_embeds, noisy_embeds)
                        display_norm(norms,context_tokens)

                        with torch.no_grad():
                            outputs = model(inputs_embeds = noisy_embeds)
                            logits = outputs.logits
                            target_score = score(logits,target)
                            contexts.append(context)
                            targets.append(target)
                            scores.append(target_score)



                i+=1

        elif context_type == 'suffix':
            suffix = '<SUF>' + ' '
            i = 0
            while i < len(uttrWords):
                context = suffix + ' '.join(uttrWords[i+1:]) + ' ' + '<MID>'
                target = uttrWords[i]

                pre_idx, suf_idx, mid_idx = extract_indices(context)
                orig_embeds = get_embeddings(model, tokenizer, context)
                context_tokens = context.split()

                # sanity check
                print(context)
                print(target)

                if A == 0:
                    print("No noise injection")
                    with torch.no_grad():
                        outputs = model(inputs_embeds = orig_embeds)
                        logits = outputs.logits
                        target_score = score(logits,target)
                        contexts.append(context)
                        targets.append(target)
                        scores.append(target_score)
                
                else:
                    if noise_type == 'temporal':
                        noisy_embeds = suffix_noise(orig_embeds, start_idx=suf_idx, end_idx=mid_idx, A=A,lam=lam)
                        norms = L2_norm(orig_embeds, noisy_embeds)
                        display_norm(norms,context_tokens)

                        with torch.no_grad():
                            outputs = model(inputs_embeds = noisy_embeds)
                            logits = outputs.logits
                            target_score = score(logits,target)
                            contexts.append(context)
                            targets.append(target)
                            scores.append(target_score)

                    elif noise_type == 'dependency':
                        current_token_id = int(uttr_df.iloc[i]["token_id"])
                        dep_dist = compute_dep_distances(uttr_graph,current_token_id)
                        lm_to_dep = lm_idx_to_dep_id(context_tokens, uttr_df)

                        noisy_embeds = suffix_noise_dep(orig_embeds,start_idx=suf_idx,end_idx=mid_idx,
                        A=A,lam=lam,lm_idx_to_dep_id=lm_to_dep,dep_dist=dep_dist)

                        norms = L2_norm(orig_embeds, noisy_embeds)
                        display_norm(norms,context_tokens)

                        with torch.no_grad():
                            outputs = model(inputs_embeds = noisy_embeds)
                            logits = outputs.logits
                            target_score = score(logits,target)
                            contexts.append(context)
                            targets.append(target)
                            scores.append(target_score)

                        
                i+=1  

    elif lm_mode == 'bidirectional':
        prefix = '<eos>' + ' ' + role + ': ' + '<PRE>' + ' '
        i = -1
        while i < len(uttrWords) - 1:
            if i == -1:
                suffix = '<SUF>' + ' ' + ' '.join(uttrWords[i+2:]) + ' ' + '<MID>'
                context = prefix + ' ' + suffix
                target = uttrWords[0]

            elif i > -1:
                prefix += uttrWords[i] + ' '
                suffix = '<SUF>' + ' ' + ' '.join(uttrWords[i+2:]) + ' ' + '<MID>'
                context = prefix + suffix 
                target = uttrWords[i+1]

            # sanity check
            print(context)
            print(target)

            pre_idx, suf_idx, mid_idx = extract_indices(context)
            orig_embeds = get_embeddings(model, tokenizer, context)

            if A == 0:
                with torch.no_grad():
                    outputs = model(inputs_embeds = orig_embeds)
                    logits = outputs.logits
                    target_score = score(logits,target)
                    contexts.append(context)
                    targets.append(target)
                    scores.append(target_score)
            
            else:
                if context_type == 'prefix':
                    if noise_type == 'temporal':
                        noisy_embeds = prefix_noise(orig_embeds, start_idx=pre_idx, end_idx=suf_idx, A=A,lam=lam)
                        norms = L2_norm(orig_embeds, noisy_embeds)
                        display_norm(norms,context_tokens)

                        with torch.no_grad():
                            outputs = model(inputs_embeds = noisy_embeds)
                            logits = outputs.logits
                            target_score = score(logits,target)
                            contexts.append(context)
                            targets.append(target)
                            scores.append(target_score)
                    
                    elif noise_type == 'dependency':
                        noisy_embeds = prefix_noise(orig_embeds, start_idx=pre_idx, end_idx=suf_idx, A=A,lam=lam)
                        norms = L2_norm(orig_embeds, noisy_embeds)
                        display_norm(norms,context_tokens)

                        with torch.no_grad():
                            outputs = model(inputs_embeds = noisy_embeds)
                            logits = outputs.logits
                            target_score = score(logits,target)
                            contexts.append(context)
                            targets.append(target)
                            scores.append(target_score)
                
                if context_type == 'suffix':
                    if noise_type == 'temporal':
                        noisy_embeds = suffix_noise(orig_embeds, start_idx=suf_idx, end_idx=mid_idx, A=A,lam=lam)
                        norms = L2_norm(orig_embeds, noisy_embeds)
                        display_norm(norms,context_tokens)

                        with torch.no_grad():
                            outputs = model(inputs_embeds = noisy_embeds)
                            logits = outputs.logits
                            target_score = score(logits,target)
                            contexts.append(context)
                            targets.append(target)
                            scores.append(target_score)
                    
                    elif noise_type == 'dependency':
                        current_token_id = int(uttr_df.iloc[i]["token_id"])
                        dep_dist = compute_dep_distances(uttr_graph,current_token_id)
                        lm_to_dep = lm_idx_to_dep_id(context_tokens, uttr_df)

                        noisy_embeds = suffix_noise_dep(orig_embeds,start_idx=suf_idx,end_idx=mid_idx,
                        A=A,lam=lam,lm_idx_to_dep_id=lm_to_dep,dep_dist=dep_dist)

                        norms = L2_norm(orig_embeds, noisy_embeds)
                        display_norm(norms,context_tokens)

                        with torch.no_grad():
                            outputs = model(inputs_embeds = noisy_embeds)
                            logits = outputs.logits
                            target_score = score(logits,target)
                            contexts.append(context)
                            targets.append(target)
                            scores.append(target_score)



            i+=1

      
subset = data[data['uttrID'].isin(subset_uttrIDs)]
subset['suffix_depNoise0.8_context'] = contexts
subset['suffix_depNoise0.8_target'] = targets
subset['suffix_depNoise0.8_score'] = scores
subset.to_csv('SWBD_DurData_depparsed100_scores.csv',index=False)

