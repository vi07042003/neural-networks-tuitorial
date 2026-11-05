import torch
import torch.nn as nn

class ZeroInitModel(nn.Module): 
    def __init__(self):
        super(ZeroInitModel, self).__init__()
        self.linear = nn.Linear(10, 2)
        nn.init.zeros_(self.linear.weight)  # Initialize weights to zero
        nn.init.zeros_(self.linear.bias)    # Initialize bias to zero
    def forward(self, x):
        return self.linear(x)
model = ZeroInitModel()
input_data = torch.randn(5, 10)  # Example input data
output_data = model(input_data)
print("Output data:", output_data)
