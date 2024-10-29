import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import tiktoken


with open('data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

enc = tiktoken.get_encoding('gpt2')

data = torch.tensor(enc.encode(text), dtype=torch.long)

n_data = int(0.9 * len(data))
train_data = data[:n_data]
test_data = data[n_data:]