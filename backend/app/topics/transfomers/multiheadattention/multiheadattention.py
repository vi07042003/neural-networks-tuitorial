
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import seaborn as sns


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.output = nn.Linear(d_model, d_model)

    def forward(self, x):

        batch_size, seq_len, _ = x.shape

        # Generate Q, K, V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Split into multiple heads
        Q = Q.view(
            batch_size, seq_len,
            self.num_heads, self.head_dim
        ).transpose(1, 2)

        K = K.view(
            batch_size, seq_len,
            self.num_heads, self.head_dim
        ).transpose(1, 2)

        V = V.view(
            batch_size, seq_len,
            self.num_heads, self.head_dim
        ).transpose(1, 2)

        # Calculate raw attention scores
        scores = Q @ K.transpose(-2, -1)

        # Scale scores
        scores = scores / math.sqrt(self.head_dim)

        # Softmax
        weights = torch.softmax(scores, dim=-1)

        # Context vector
        context = weights @ V

        # Combine heads
        context = context.transpose(1, 2).contiguous()

        context = context.view(
            batch_size, seq_len, self.d_model
        )

        # Output projection
        output = self.output(context)

        return output, scores, weights


# -----------------------------
# Example
# -----------------------------

d_model = 8
num_heads = 4

tokens = ["I", "love", "AI"]

x = torch.randn(1, 3, d_model)

model = MultiHeadAttention(d_model, num_heads)

output, scores, attention_weights = model(x)

print("Input:", x.shape)
print("Output:", output.shape)
print("Attention Scores:", scores.shape)
print("Attention Weights:", attention_weights.shape)


# -----------------------------
# Visualization
# -----------------------------

# Convert tensors into NumPy
scores_np = scores[0].detach().numpy()
weights_np = attention_weights[0].detach().numpy()

# Create 3 visualization sections
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# 1. Raw attention scores (Head 1)
im1 = axes[0, 0].imshow(
    scores_np[0],
    cmap="viridis"
)

axes[0, 0].set_title("Raw Attention Scores - Head 1")
axes[0, 0].set_xticks(range(len(tokens)))
axes[0, 0].set_yticks(range(len(tokens)))
axes[0, 0].set_xticklabels(tokens)
axes[0, 0].set_yticklabels(tokens)
axes[0, 0].set_xlabel("Key Tokens")
axes[0, 0].set_ylabel("Query Tokens")
fig.colorbar(im1, ax=axes[0, 0])


# 2. Softmax attention weights (Head 1)
im2 = axes[0, 1].imshow(
    weights_np[0],
    cmap="Blues",
    vmin=0,
    vmax=1
)

axes[0, 1].set_title("Attention Weights - Head 1")
axes[0, 1].set_xticks(range(len(tokens)))
axes[0, 1].set_yticks(range(len(tokens)))
axes[0, 1].set_xticklabels(tokens)
axes[0, 1].set_yticklabels(tokens)
axes[0, 1].set_xlabel("Key Tokens")
axes[0, 1].set_ylabel("Query Tokens")

# Display numerical values
for i in range(len(tokens)):
    for j in range(len(tokens)):
        axes[0, 1].text(
            j, i,
            f"{weights_np[0][i, j]:.2f}",
            ha="center",
            va="center"
        )

fig.colorbar(im2, ax=axes[0, 1])


# 3. Compare all four attention heads
for head in range(num_heads):

    ax = axes[1, 0] if head == 0 else None

    if head == 0:
        axes[1, 0].plot(
            tokens,
            weights_np[head][0],
            marker="o",
            label="Head 1"
        )

        axes[1, 0].plot(
            tokens,
            weights_np[head][1],
            marker="o",
            label="Head 1 - Query 2"
        )

    else:
        axes[1, 0].plot(
            tokens,
            weights_np[head][0],
            marker="o",
            label=f"Head {head + 1}"
        )

axes[1, 0].set_title("Attention Comparison Across Heads")
axes[1, 0].set_xlabel("Key Tokens")
axes[1, 0].set_ylabel("Attention Weight")
axes[1, 0].legend()
axes[1, 0].grid()


# 4. Average attention across all heads
average_weights = weights_np.mean(axis=0)

im4 = axes[1, 1].imshow(
    average_weights,
    cmap="plasma",
    vmin=0,
    vmax=1
)

axes[1, 1].set_title("Average Attention Across Heads")
axes[1, 1].set_xticks(range(len(tokens)))
axes[1, 1].set_yticks(range(len(tokens)))
axes[1, 1].set_xticklabels(tokens)
axes[1, 1].set_yticklabels(tokens)

fig.colorbar(im4, ax=axes[1, 1])

plt.tight_layout()
plt.show()
