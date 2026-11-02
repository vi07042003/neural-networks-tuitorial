import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms ,datasets
from torch.utils.data import DataLoader ,Subset

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)   

train_dataset = Subset(train_dataset, range(5000))


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 64)
        self.bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = Model()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),
                           lr=0.001)
for epoch in range(2):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            print(f"Epoch [{epoch + 1}/2], Loss: {running_loss / len(train_loader):.4f}")