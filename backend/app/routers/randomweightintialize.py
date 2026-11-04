import torch
import torch.nn as nn

layer = nn.Linear(10, 2)

nn.init.normal_(layer.weight, mean=0.0, std=1.0)  # Initialize weights with normal distribution
nn.init.zeros_(layer.bias)  # Initialize bias with zeros


print("Initialized weights:", layer.weight)
print("Initialized bias:", layer.bias)
