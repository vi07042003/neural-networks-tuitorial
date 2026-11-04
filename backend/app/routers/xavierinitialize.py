import torch
import torch.nn as nn

layer = nn.Linear(10, 2)

nn.init.xavier_normal_(layer.weight)  # Initialize weights with Xavier normal distribution
nn.init.zeros_(layer.bias)  # Initialize bias with zeros

print("Initialized weights:", layer.weight)
print("Initialized bias:", layer.bias)
