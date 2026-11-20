training_data = []
window_size = 1

sentences = [
    "i love machine learning",
    "i love deep learning",
    "machine learning is interesting",
    "deep learning uses neural networks",
    "i love artificial intelligence",
]

# 1. Build vocabulary and index mappings
vocab = sorted({word for sentence in sentences for word in sentence.split()})
word_to_index = {word: idx for idx, word in enumerate(vocab)}
index_to_word = {idx: word for word, idx in word_to_index.items()}

# 2. Generate (target_idx, context_idx) Skip-gram pairs
for sentence in sentences:
    tokens = sentence.split()

    for i, target_word in enumerate(tokens):
        target = word_to_index[target_word]

        for j in range(
            max(0, i - window_size),
            min(len(tokens), i + window_size + 1),
        ):
            if i != j:
                context = word_to_index[tokens[j]]
                training_data.append((target, context))

print(f"Vocabulary size: {len(vocab)}")
print(f"Total training pairs: {len(training_data)}")
print("Sample pairs (target_idx, context_idx):", training_data[:5])
print(
    "Sample pairs (words):",
    [(index_to_word[t], index_to_word[c]) for t, c in training_data[:5]],
)