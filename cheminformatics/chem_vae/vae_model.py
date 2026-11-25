import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np

#test variable size conv

class FullyConvNN(nn.Module):
  def __init__(self, embedding_dim, vocab_size, hidden_size, out_size):
    super().__init__()
    self.embed_dim = embedding_dim
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.out_size = out_size

    self.embed = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)

    self.l1 = nn.Conv1d(in_channels = self.embed_dim, out_channels = hidden_size*2, kernel_size = 3, dilation=1)
    self.l2 = nn.Conv1d(in_channels = hidden_size*2, out_channels = hidden_size, kernel_size = 3, dilation=1)
    self.l3 = nn.Conv1d(in_channels = hidden_size, out_channels = hidden_size, kernel_size = 5, dilation=2)
    self.l4 = nn.Conv1d(in_channels = hidden_size, out_channels = hidden_size, kernel_size = 7, dilation=2)

    self.linear_blowup = nn.Linear(in_features = self.hidden_size, out_features = self.out_size)
  
  def inference(self, data):
    relu = nn.ReLU()
    data = pad_sequence(data, batch_first=True, padding_value=0)
    data = torch.permute(data, (0,2,1)) #embedding to outer

    x = relu(self.l1(data))
    print(x.shape)
    x = relu((self.l2(x)))
    print(x.shape)
    x = relu(self.l3(x))
    print(x.shape)
    x = relu(self.l4(x))
    print(x.shape)


    data = torch.mean(x, dim=2)
    data = self.linear_blowup(data)
    return data

def MultiLayerGRU(nn.Module):
  def __init__(self, embedding_dim, vocab_size, hidden_size, out_size):
    super().__init__()
    self.gru = nn.GRU(
      input_size=embedding_dim,
      hidden_size=hidden_size,
      num_layers=2,
      batch_first=True
    )
    # out shape: (batch_size, sequence_length, hidden_size) -> outputs from layer 3
    # hn shape: (num_layers, batch_size, hidden_size) -> final hidden state for each layer
    # gru generates next token and decodes, autoregressively