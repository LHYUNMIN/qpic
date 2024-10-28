import torch
from torch import nn
# Function to add 1D positional embeddings
def add_1d_positional_embeddings(x):
    _, _, H, W = x.size()
    pos_h = torch.arange(0, H, dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(x.device)
    pos_w = torch.arange(0, W, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(x.device)
    
    # Normalize positional encodings
    pos_h /= H
    pos_w /= W
    
    # Create 1D positional embeddings and concatenate with input
    pos_encodings_h = pos_h.expand(-1, -1, W).unsqueeze(0)
    pos_encodings_w = pos_w.expand(-1, H, -1).unsqueeze(0)
    x = torch.cat([x, pos_encodings_h, pos_encodings_w], dim=1)
    return x
