import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from sentence_transformers import SentenceTransformer # Sentence encoder

import pandas as pd
import numpy as np
from itertools import combinations
from numpy.linalg import norm
from PIL import Image

class Towers(nn.Module):
  
  def __init__(self, num_classes = 3, image_weight = 0.5, text_weight = 0.5):
    super(Towers, self).__init__()

    img_layers = self.LinearBlock(64, 512) + sum([self.LinearBlock(512, 512) for i in range(4)], [])
    text_layers = self.LinearBlock(3072, 512) + self.LinearBlock(512, 512)
    
    self.downsize = nn.Sequential(*self.LinearBlock(2048, 64, 0.0))
    self.img_features = nn.Sequential(*img_layers)
    self.text_features = nn.Sequential(*text_layers)
    
    self.shared = nn.Linear(512, num_classes)
    self.batchnorm = nn.BatchNorm1d(512)

    self.image_weight = image_weight
    self.text_weight = text_weight
  
  def LinearBlock(self, in_features, out_features, dropout_p = 0.15):
    block = [nn.Linear(in_features, out_features), nn.BatchNorm1d(out_features), nn.LeakyReLU(0.1), nn.Dropout(dropout_p)]  # I have also used BatchNorm after each layer
    return block

  def forward(self, img_embedding, text_embedding): #  takes the image and text embedding as input

    img_f = self.img_features(self.downsize(img_embedding)) # finds image features
    text_f = self.text_features(text_embedding) # finds text features

    # L2 normalisation of image features
    img_f_norm = torch.norm(img_f, p=2, dim=1).detach() 
    img_f = img_f.div(img_f_norm.unsqueeze(1))
    
    # L2 normalisation of text features
    text_f_norm = torch.norm(text_f, p=2, dim=1).detach()
    text_f = text_f.div(text_f_norm.unsqueeze(1))

    image_shared_output = self.shared(img_f)
    text_shared_output = self.shared(text_f)
    shared_output = (image_shared_output * self.image_weight) + (text_shared_output * self.text_weight)
    
    return shared_output, image_shared_output, text_shared_output 

class ImageTextDataset(Dataset):
    # in this implementation, the images are present in ./images/*.jpg
    def __init__(self, X, y, transforms = None): 
      self.filenames = X[:, 0]
      self.targets = y
      self.text = X[:, 1]
      self.data_len = len(self.filenames)
      self.transforms = transforms
    
    def __getitem__(self, index):
      filename = self.filenames[index]
      img = Image.open(filename)
      img = self.transforms(img)
      target = self.targets[index]
      text = self.text[index]
      return (img, text, target)

    def __len__(self):
      return self.data_len

def return_adj_matrix(dataframe):
  
  classes = sorted(dataframe.processed_classes.unique())
  classes_dict = {v:k for (k,v) in enumerate(classes)}
  class_combinations = list(combinations(np.arange(0, len(classes)), r = 2))

  sent_bert = SentenceTransformer('bert-base-nli-mean-tokens').eval()
  sentence_embeddings = sent_bert.encode(classes)

  adj_matrix =np.zeros((len(sentence_embeddings), len(sentence_embeddings)))
  normalised_sentence_embeddings = [i/norm(i) for i in sentence_embeddings]
  for class_tuple in class_combinations:
    u, v = class_tuple[0], class_tuple[1]
    adj_matrix[class_tuple] = adj_matrix[(v,u)] = 1 - sum(normalised_sentence_embeddings[u] * normalised_sentence_embeddings[v])

  return adj_matrix