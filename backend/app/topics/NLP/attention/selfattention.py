
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np


class SelfAttention(nn.Module):

    def __init__(self, d_model):
        super().__init__()

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

    def forward(self, x):

        # Generate Q, K and V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Calculate attention scores
        scores = Q @ K.transpose(-2, -1)

        # Scale attention scores
        scores = scores / math.sqrt(K.size(-1))

        # Convert scores into attention weights
        weights = torch.softmax(scores, dim=-1)

        # Calculate weighted sum of values
        output = weights @ V

        return output, weights


# Example input
torch.manual_seed(42)

batch_size = 1
sequence_length = 4
d_model = 8

x = torch.randn(batch_size, sequence_length, d_model)

model = SelfAttention(d_model)

# Forward pass
with torch.no_grad():
    output, attention_weights = model(x)


# Convert tensors to NumPy
input_data = x[0].numpy()
output_data = output[0].numpy()
attention_data = attention_weights[0].numpy()


# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))


# 1. Attention Heatmap
im1 = axes[0, 0].imshow(
    attention_data,
    cmap="viridis",
    aspect="auto"
)

axes[0, 0].set_title("Self-Attention Heatmap")
axes[0, 0].set_xlabel("Key Positions")
axes[0, 0].set_ylabel("Query Positions")

axes[0, 0].set_xticks(range(sequence_length))
axes[0, 0].set_yticks(range(sequence_length))

for i in range(sequence_length):
    for j in range(sequence_length):
        axes[0, 0].text(
            j, i,
            f"{attention_data[i, j]:.2f}",
            ha="center",
            va="center",
            color="white"
        )

fig.colorbar(im1, ax=axes[0, 0])


# 2. Attention Weights Line Graph
for i in range(sequence_length):
    axes[0, 1].plot(
        range(sequence_length),
        attention_data[i],
        marker="o",
        label=f"Query {i}"
    )

axes[0, 1].set_title("Attention Distribution")
axes[0, 1].set_xlabel("Key Positions")
axes[0, 1].set_ylabel("Attention Weight")
axes[0, 1].set_xticks(range(sequence_length))
axes[0, 1].legend()
axes[0, 1].grid(True)


# 3. Input Embeddings
im2 = axes[1, 0].imshow(
    input_data,
    cmap="coolwarm",
    aspect="auto"
)

axes[1, 0].set_title("Input Embeddings")
axes[1, 0].set_xlabel("Embedding Dimensions")
axes[1, 0].set_ylabel("Sequence Positions")
fig.colorbar(im2, ax=axes[1, 0])


# 4. Output Embeddings
im3 = axes[1, 1].imshow(
    output_data,
    cmap="coolwarm",
    aspect="auto"
)

axes[1, 1].set_title("Output Embeddings After Attention")
axes[1, 1].set_xlabel("Embedding Dimensions")
axes[1, 1].set_ylabel("Sequence Positions")
fig.colorbar(im3, ax=axes[1, 1])


plt.tight_layout()
plt.show()


# Print shapes
print("Input shape:", x.shape)
print("Attention weights shape:", attention_weights.shape)
print("Output shape:", output.shape)

# Check if each attention row sums to 1
print("Attention row sums:", attention_weights.sum(dim=-1))
