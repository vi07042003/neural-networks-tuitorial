import torch 
import torch.nn as nn

class HeInitModel(nn.Module):
    def __init__(self):
        super(HeInitModel, self).__init__()
        self.linear = nn.Linear(7, 2)
        nn.init.kaiming_normal_(self.linear.weight, mode='fan_in', nonlinearity='relu')  # He initialization for weights
        nn.init.zeros_(self.linear.bias)  # Initialize bias to zero

    def forward(self, x):
        return self.linear(x)


model = HeInitModel()
input_data = torch.randn(5, 7)  # Example input data
output_data = model(input_data)
print("Output data:", output_data)