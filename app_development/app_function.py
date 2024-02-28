#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import esm
import torch

def esm_embedding(excel_path):
    excel_path = excel_path
    dataset = pd.read_excel(excel_path, na_filter=False)
    sequence_list = dataset['Sequence']
    peptide_sequence_list = []
    for seq in sequence_list:
        format_seq = [seq,seq]
        tuple_sequence = tuple(format_seq)
        peptide_sequence_list.append(tuple_sequence)
    #load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    
    #load data
    data = peptide_sequence_list
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    
    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[6], return_contacts=True)
    token_representations = results["representations"][6]
    
    # Generate per-sequence representations via averaging
    sequence_representations = []
    for i, token_len in enumerate(batch_lens):
        each_seq_rep = token_representations[i, 1:token_len - 1].mean(0).tolist()  
        sequence_representations.append(each_seq_rep)

    embedding_results = pd.DataFrame(sequence_representations)
    return embedding_results


# In[ ]:




