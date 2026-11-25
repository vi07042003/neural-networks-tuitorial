
import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt
import numpy as np


# ==========================================
# 1. POSITIONAL ENCODING
# ==========================================

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=100):

        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(
            0, max_len
        ).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):

        return x + self.pe[:, :x.size(1), :]


# ==========================================
# 2. SELF ATTENTION
# ==========================================


class SelfAttention(nn.Module):

    def __init__(self, d_model):
        super().__init__()

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):

        # Generate Q, K and V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Calculate attention scores
        scores = Q @ K.transpose(-2, -1)

        # Scale attention scores
        scores = scores / math.sqrt(K.size(-1))

        # Apply attention mask
        if mask is not None:
            scores = scores.masked_fill(
                mask == 0,
                float('-inf')
            )

        # Convert scores into attention weights
        weights = torch.softmax(scores, dim=-1)

        # Calculate weighted sum of values
        output = weights @ V

        return output, weights



# ==========================================
# 3. COMPLETE TRANSFORMER MODEL
# ==========================================

class TransformerModel(nn.Module):

    def __init__(self, d_model):

        super().__init__()

        self.position = PositionalEncoding(d_model)

        self.attention = SelfAttention(d_model)

        self.output_layer = nn.Linear(d_model, d_model)

    def forward(self, x):

        # Add positional information
        x = self.position(x)

        # Self attention
        output, weights = self.attention(x)

        # Final output
        output = self.output_layer(output)

        return output, weights


# ==========================================
# 4. INPUT DATA
# ==========================================

torch.manual_seed(42)

batch_size = 1
sequence_length = 4
d_model = 8

x = torch.randn(
    batch_size,
    sequence_length,
    d_model
)

# Target output (example training target)
target = torch.randn(
    batch_size,
    sequence_length,
    d_model
)


# ==========================================
# 5. INITIALIZE MODEL
# ==========================================

model = TransformerModel(d_model)

# Loss function
criterion = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# ==========================================
# 6. BACKPROPAGATION / TRAINING LOOP
# ==========================================

epochs = 100

loss_history = []

for epoch in range(epochs):

    # Forward pass
    output, attention_weights = model(x)

    # Calculate loss
    loss = criterion(output, target)

    # Clear old gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Update weights
    optimizer.step()

    loss_history.append(loss.item())

    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {loss.item():.6f}"
        )


# ==========================================
# 7. FINAL FORWARD PASS
# ==========================================

model.eval()

with torch.no_grad():

    final_output, attention_weights = model(x)


# ==========================================
# 8. CONVERT TO NUMPY
# ==========================================

input_data = x[0].numpy()

output_data = final_output[0].numpy()

attention_data = attention_weights[0].numpy()

position_data = model.position.pe[0, :sequence_length].numpy()


# ==========================================
# 9. VISUALIZATION
# ==========================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))


# Attention heatmap
im1 = axes[0, 0].imshow(
    attention_data,
    cmap="viridis",
    aspect="auto"
)

axes[0, 0].set_title("Self-Attention Heatmap")
axes[0, 0].set_xlabel("Key Positions")
axes[0, 0].set_ylabel("Query Positions")

fig.colorbar(im1, ax=axes[0, 0])


# Input embeddings
im2 = axes[0, 1].imshow(
    input_data,
    cmap="coolwarm",
    aspect="auto"
)

axes[0, 1].set_title("Original Input Embeddings")
axes[0, 1].set_xlabel("Embedding Dimensions")
axes[0, 1].set_ylabel("Sequence Positions")

fig.colorbar(im2, ax=axes[0, 1])


# Positional encoding
im3 = axes[0, 2].imshow(
    position_data,
    cmap="coolwarm",
    aspect="auto"
)

axes[0, 2].set_title("Positional Encoding")
axes[0, 2].set_xlabel("Embedding Dimensions")
axes[0, 2].set_ylabel("Sequence Positions")

fig.colorbar(im3, ax=axes[0, 2])


# Output embeddings
im4 = axes[1, 0].imshow(
    output_data,
    cmap="coolwarm",
    aspect="auto"
)

axes[1, 0].set_title("Output After Attention")
axes[1, 0].set_xlabel("Embedding Dimensions")
axes[1, 0].set_ylabel("Sequence Positions")

fig.colorbar(im4, ax=axes[1, 0])


# Loss graph
axes[1, 1].plot(loss_history)

axes[1, 1].set_title("Training Loss")
axes[1, 1].set_xlabel("Epochs")
axes[1, 1].set_ylabel("MSE Loss")
axes[1, 1].grid(True)


# Attention distribution
for i in range(sequence_length):

    axes[1, 2].plot(
        range(sequence_length),
        attention_data[i],
        marker="o",
        label=f"Query {i}"
    )

axes[1, 2].set_title("Attention Distribution")
axes[1, 2].set_xlabel("Key Positions")
axes[1, 2].set_ylabel("Attention Weight")
axes[1, 2].legend()
axes[1, 2].grid(True)


plt.tight_layout()
plt.show()


# ==========================================
# 10. PRINT SHAPES
# ==========================================

print("Input shape:", x.shape)

print("Positional encoding shape:", position_data.shape)

print("Attention weights shape:", attention_weights.shape)

print("Output shape:", final_output.shape)

print(
    "Attention row sums:",
    attention_weights.sum(dim=-1)
)
