import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden = nn.Linear(2, 2)

        self.gelu = nn.GELU()

        self.output = nn.Linear(2, 1)

        with torch.no_grad():
            self.hidden.weight.copy_(
                torch.tensor([
                    [0.5, 0.2],   
                    [-0.4, 0.3] 
                ])
            )

            self.hidden.bias.copy_(
                torch.tensor([0.1, 0.1])
            )

            self.output.weight.copy_(
                torch.tensor([[0.6, 0.8]])
            )

            self.output.bias.copy_(
                torch.tensor([0.1])
            )

    def forward(self, x):
        z = self.hidden(x)
        a = self.gelu(z)
        y_hat = self.output(a)

        return y_hat

x = torch.tensor([[2.0, 3.0]])
y = torch.tensor([[2.0]])

model = NeuralNetwork()

y_hat = model(x)


loss = 0.5 * ((y_hat - y) ** 2).mean()

print("Prediction:", y_hat.item())
print("Loss:", loss.item())


loss.backward()

for name, parameter in model.named_parameters():
    print(f"{name} gradient:\n{parameter.grad}")