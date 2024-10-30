import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import tiktoken

block_size = 24  # max text chunk length
batch_size = 64  # number of chunks processed at same time

with open('../data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

enc = tiktoken.get_encoding('gpt2')
vocab_size = enc.n_vocab

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

data = torch.tensor(enc.encode(text), dtype=torch.long)

n_data = int(0.9 * len(data))
train_data = data[:n_data]
test_data = data[n_data:]


def get_batch(split):  # generate targets chunk and input chunks
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])  # INPUT
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])  # TARGET
    x, y = x.to(device), y.to(device)
    return x, y


eval = 200


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval)
        for k in range(eval):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Net(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding_dim = 384
        self.token_embeddings = nn.Embedding(vocab_size, self.embedding_dim)
        # projection layer to map embeddings back to vocab size
        self.lm_head = nn.Linear(self.embedding_dim, vocab_size)

        # init weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # get embeddings
        embeddings = self.token_embeddings(idx)  #(B,T,embedding_dim)
        # project to vocab size
        logits = self.lm_head(embeddings)  #(B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = Net(vocab_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

interval = 300
for t in range(3000):
    if t % interval == 0:
        losses = estimate_loss()
        print(f"epoch {t}: train loss {losses['train']:.4f}, test loss {losses['test']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(enc.decode(model.generate(context, max_new_tokens=500)[0].tolist()))