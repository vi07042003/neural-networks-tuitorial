import torch
import torch.nn as nn

text = "cat is drinking water"

## tokenization
tokens= text.lower().split()

## create vocabulary
vocab = {word: i for i , word in enumerate(set(tokens))}

## converting tokens into ID
token_id = [vocab[word]for word in tokens]

print("tokens",tokens)
print("vocab",vocab)
print("token_id",token_id)


#### embedding layer

vocab_size = len(vocab)

embeddings_dim  = 4

embeddings = nn.Embedding(
    num_embeddings=vocab_size ,
    embedding_dim= embeddings_dim
)

input_ids = torch.tensor(token_id)


## convert token_id into vectors

output = embeddings(input_ids)

print(output)


print("output_shape",output.shape)