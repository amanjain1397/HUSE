from itertools import combinations

import torch
import torch.nn.functional as F
import torch.nn as nn
from pytorch_pretrained_bert import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

cl_loss_type = nn.CrossEntropyLoss()
gr_loss_type = nn.MSELoss()

def get_encoding(text, bert, device):
    tokenized_text = tokenizer.tokenize(text) # First step involves the tokenisation of the string. It expects that the string starts and ends with [CLS] [SEP] tags.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text) # Converting tokens to ids in the BERT vocab
    segments_ids = [1] * len(tokenized_text)

    tokens_tensor = torch.tensor([indexed_tokens]).to(device)
    segments_tensors = torch.tensor([segments_ids]).to(device)

    with torch.no_grad():
      encoded_layers, _ = bert(tokens_tensor, segments_tensors) # results a total of 12 layers, but we only need the last 4 layers.

    token_embeddings = torch.stack(encoded_layers, dim=0).squeeze(1) 
    token_embeddings = token_embeddings.permute(1,0,2) # Correcting the dimensions

    token_vecs_cat = []
    for token in token_embeddings:
        cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0) # Getting the embeddings only from the last 4 layers
        token_vecs_cat.append(cat_vec)
    
    return torch.mean(torch.stack(token_vecs_cat, dim = 0), dim = 0) # returning the mean of the last 4 layer embeddings 
  
def graph_loss_utility(outputs, labels, adj_matrix, device, margin = 0.7):

  x_tuples = list(combinations(range(outputs.shape[0]), r = 2)) # [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] for a given batch_size = 4 --> 4C2 = 6, hence 6 tuples
  target_tuples = [tuple(labels[[*tuple_]].detach().cpu().numpy()) for tuple_ in x_tuples] # [(35, 41), (35, 11), (35, 47), (41, 11), (41, 47), (11, 47)] --> 6 tuples here too.

  # Using target_tuples, we find the cosine distance values between these classes using the adjacency matrix -- > A_ij
  A_ij = torch.tensor([adj_matrix[tuple_] for tuple_ in target_tuples]).to(device)
  
  # Using x_tuples and outputs, we find the cosine distance between two points in each tuple corresponding to real target index
  cosine_x = torch.stack([F.cosine_similarity(outputs[tuple_[0]], outputs[tuple_[1]], dim = 0, ) for tuple_ in x_tuples])

  sigma_x = torch.tensor([1 if ((A_ij[i] < margin) and (cosine_x[i] < margin)) else 0 for i in range(len(A_ij)) ]).to(device)
  
  return (A_ij * sigma_x).float(), cosine_x * sigma_x

def classification_loss_fn(outputs, labels):
  return cl_loss_type(outputs, labels)

def graph_loss_fn(outputs, labels, adj_matrix, device): 
  g1, g2 = graph_loss_utility(outputs, labels, adj_matrix, device)
  return gr_loss_type(g2, g1)/len(g1)

def gap_loss_fn(image_features, text_features): 
  return torch.mean(1 - F.cosine_similarity(image_features, text_features))