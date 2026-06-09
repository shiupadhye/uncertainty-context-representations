import re
import torch
import spacy
import stanza
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.serialization import add_safe_globals


add_safe_globals([
    np.core.multiarray._reconstruct,
    np.ndarray,
    np.dtype,
])

_orig_torch_load = torch.load

def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)

torch.load = _patched_load


def dep_parse(orig_tokens):
    """
    Generates dependency-parse for sentence 
    
    :param orig_tokens: List of sentence tokens
    """
    # Run Stanza with pretokenized input
    doc = nlp([orig_tokens])          
    sent = doc.sentences[0]
    
    word_to_token = {}
    for token_id, tok in enumerate(sent.tokens):
        for w in tok.words:
            word_to_token[w.id] = token_id

    rows = []

    for token_id, tok in enumerate(sent.tokens):
        word_ids_in_token = {w.id for w in tok.words}

        main_word = None
        for w in tok.words:
            if w.head not in word_ids_in_token:
                main_word = w
                break
        if main_word is None:
            main_word = tok.words[0]

        if main_word.head == 0:
            head_token_id = -1   # ROOT
        else:
            head_token_id = word_to_token[main_word.head]

        rows.append({
            "stanza_token": tok.text,
            "token_id": token_id,
            "head_token_id": head_token_id,
            "dep_rel": main_word.deprel
        })

    return rows




# stanza tokenizer
nlp = stanza.Pipeline("en", processors="tokenize,pos,lemma,depparse", tokenize_pretokenized=True,verbose=False)

data = pd.read_csv('SWBD_durationData.csv')
data.head()


uttrIDs = np.unique(data['uttrID'].values)
stanza_tokens = []
token_ids = []
head_token_ids = []
deprel_tags = []
for uttrID in tqdm(uttrIDs):
    uttrData = data[data['uttrID'] == uttrID]
    uttrWords = list(uttrData['word'].values)
    parsed_output = dep_parse(uttrWords)
    # save
    for i in range(len(parsed_output)):
        row = parsed_output[i]
        stanza_tokens.append(row['stanza_token'])
        token_ids.append(row['token_id'])
        head_token_ids.append(row['head_token_id'])
        deprel_tags.append(row['dep_rel'])
    
    

subset = data[data['uttrID'].isin(uttrIDs)]
subset['stanza_token'] = stanza_tokens
subset['token_id'] = token_ids
subset['head_token_id'] = head_token_ids
subset['dep_rel'] = deprel_tags
subset.to_csv('SWBD_durationData_depparsed.csv',index=False)
