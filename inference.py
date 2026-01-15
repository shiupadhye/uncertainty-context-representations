import math
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

TOKENIZER_PATH = "/home/shiva/PROJECTS/word-infill-model-training/tokenizer/candor_tokenizer.json"
MODEL_PATH = "/home/shiva/PROJECTS/word-infill-model-training/models/gpt2-fw-candor/checkpoint-662895"

def score(logits, token_id):
    """
    Compute log p(token | context) from AR LM logits.
    """
    last_logits = logits[0, -1]          
    log_probs = F.log_softmax(last_logits, dim=-1)
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

def inject_noise_in_prefix(embeds,start_idx,end_idx,A,lam):
    """
    Exponential decay in prefix
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

def inject_noise_in_suffix(embeds,start_idx,end_idx,A,lam):
    """
    Exponential decay in suffix
    """
    noisy_embeds = embeds.clone()
    _, seq_len, D = embeds.shape

    for i in range(start_idx, end_idx):
        # recency of token
        distance = i - start_idx
        # A * exp{-lambda * distance}
        sigma = A * (1 - math.exp(-lam * distance))
        # generate noise and decay noise 
        noisy_embeds[0, i,:] += torch.randn(D, device=embeds.device) * sigma

    return noisy_embeds


def prepare_noisy_prefix(model,tokenizer,context,A,lam,infill=False):
    """
    Injects noise into the prefix
    """
    context_tokens = context.split()
    if infill:
        start_idx = 3
        end_idx = context_tokens.index("<MID>")
    else:
        start_idx = 2
        end_idx = len(context_tokens)
    
    embeds = get_embeddings(model,tokenizer,context)    
    if A == 0.0:
        return embeds
    else:
        modified_embeds = inject_noise_in_prefix(embeds,start_idx,end_idx,A,lam)
        return modified_embeds

def prepare_noisy_suffix(embeds,start_idx,end_idx,A,lam,infill):
    """
    Injects noise into the suffix
    """
    context_tokens = context.split()
    if infill:
        start_idx = 1
        end_idx = context_tokens.index("<MID>")
    else:
        start_idx = 2
        end_idx = len(context_tokens)
    
    embeds = get_embeddings(model,tokenizer,context)   
    if A == 0.0:
         return embeds
    else:
        modified_embeds = inject_noise_in_prefix(embeds,start_idx,end_idx,A,lam)
        return modified_embeds


tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)


data = pd.read_csv('SWBD_durData_fluent.csv')
data.head()

infill = True
direction = "backward"
A = 1.0
lam = 0.5

print("----------------------------")

print("Direction: %s" % direction)
print("Infill: %s" % str(infill))
print("A = %f" % A)
print("lam = %f" % lam)

print("----------------------------")


contexts = []
target_scores = []
uttrIDs = np.unique(data['uttrID'].values)
subset_uttrIDs = uttrIDs[:1000]

for uttrID in subset_uttrIDs:
    uttrData = data[data['uttrID'] == uttrID]
    spID = uttrData['spID'].values[0].split("_")[-1]
    uWordIDs = list(uttrData['uWordID'].values)
    uttrWords = list(uttrData['word'].values)

    if direction == "forward":
        if infill:
            prefix = "<eos>" + " " + spID + ": " + "<PRE>" + " " 
        else:
            prefix = "<eos>" + " " + spID + ": "


        i = -1
        while i < len(uttrWords) - 1:
            if i == -1:
                if infill:
                    context = prefix + "<MID>"
                else:
                    context = prefix
                target = uttrWords[0]

            elif i > -1:
                prefix += uttrWords[i] + " " 
                if infill:
                    context = prefix + "<MID>"
                else:
                    context = prefix

                target = uttrWords[i+1]

            # sanity check
            print(context)
            print(target)
            embeds = prepare_noisy_prefix(model, tokenizer, context, A, lam, infill)
            target_id = tokenizer.encode(target)[0]

            with torch.no_grad():
                outputs = model(inputs_embeds = embeds)
                logits = outputs.logits
                target_score = score(logits,target_id)
                # store
                contexts.append(context)
                target_scores.append(target_score)
                
            i+=1
    
    elif direction == 'backward' and infill == False:
        rev_target_scores = []
        rev_contexts = []
        rev_uttrWords = uttrWords[::-1]
        prefix = "<eos>" + " " + spID + ": "

        i = -1
        while i < len(rev_uttrWords) - 1:

            if i == -1:
                context = prefix
                target = rev_uttrWords[0]

            elif i > -1:
                prefix += rev_uttrWords[i] + " "
                context = prefix
                target = rev_uttrWords[i + 1]

            # sanity check
            print(context)
            print(target)

            embeds = prepare_noisy_prefix(model, tokenizer, context, A, lam, infill)
            target_id = tokenizer.encode(target)[0]

            with torch.no_grad():
                outputs = model(inputs_embeds=embeds)
                logits = outputs.logits
                target_score = score(logits, target_id)

                rev_contexts.append(context)
                rev_target_scores.append(target_score)
            

            i+=1
    
        rev_contexts = rev_contexts[::-1]
        rev_target_scores = rev_target_scores[::-1]

        contexts.extend(rev_contexts)
        target_scores.extend(rev_target_scores)

    elif direction == 'backward' and infill == True:
        rev_target_scores = []
        rev_contexts = []

        i = 0
        while i < len(uttrWords):
            suffix = " ".join(uttrWords[i+1:])
            context = "<SUF>" + " " + suffix + " " + "<MID>"
            target = uttrWords[i]

            print(context)
            print(target)

            embeds = prepare_noisy_suffix(model,tokenizer,context,A,lam,infill)
            target_id = tokenizer.encode(target)[0]

            with torch.no_grad():
                outputs = model(inputs_embeds = embeds)
                logits = outputs.logits
                target_score = score(logits,target_id)
                # store
                contexts.append(context)
                target_scores.append(target_score)

            i += 1


subset = data[data['uttrID'].isin(subset_uttrIDs)]
subset['bwInfillContext_lam0.5'] = contexts
subset['bwInfillPred_lam0.5'] = target_scores
subset.to_csv('SWBD_durData_bwInfill_lam0.5.csv',index=False)
    
                
                
                



            






