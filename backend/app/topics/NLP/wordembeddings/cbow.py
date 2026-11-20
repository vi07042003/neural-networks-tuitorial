import torch
import torch.nn as nn
import torch.optim as optim

sentences = [
    "i love machine learning",
    "i love deep learning",
    "machine learning is interesting",
    "deep learning uses neural networks",
    "i love artificial intelligence",
]

# Creating vocabulary
words = set()
for sentence in sentences:
    words.update(sentence.split())
vocabulary = sorted(words)

word_to_index = {word: index for index, word in enumerate(vocabulary)}
index_to_word = {index: word for word, index in word_to_index.items()}
vocab_size = len(vocabulary)

print("Vocabulary:", word_to_index)
print("Vocabulary Size:", vocab_size)

# CBOW: predict target_word from surrounding context_words
training_data = []
for sentence in sentences:
    tokens = sentence.split()
    for i in range(len(tokens)):
        target_word = tokens[i]
        context_words = []
        if i > 0:
            context_words.append(tokens[i - 1])
        if i < len(tokens) - 1:
            context_words.append(tokens[i + 1])
        if context_words:
            training_data.append((context_words, target_word))

print("Training Data:", training_data)


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context):
        # context shape: (batch_size, context_len)
        embeddings = self.embedding(context)      # (batch_size, context_len, embedding_dim)
        hidden = embeddings.mean(dim=1)           # (batch_size, embedding_dim)
        output = self.linear(hidden)              # (batch_size, vocab_size)
        return output


model = CBOW(vocab_size=vocab_size, embedding_dim=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(300):
    total_loss = 0.0

    for context, target in training_data:
        context_indices = [word_to_index[w] for w in context]
        target_index = word_to_index[target]

        context_tensor = torch.tensor([context_indices], dtype=torch.long)  # (1, context_len)
        target_tensor = torch.tensor([target_index], dtype=torch.long)      # (1,)

        optimizer.zero_grad()
        output = model(context_tensor)
        loss = criterion(output, target_tensor)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

for word in vocabulary:
    index = word_to_index[word]
    vector = model.embedding.weight[index]
    print(word, vector.detach().numpy())