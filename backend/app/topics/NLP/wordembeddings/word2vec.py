sentences = [
    ["i", "love", "machine", "learning"],
    ["i", "love", "deep", "learning"],
    ["machine", "learning", "is", "interesting"],
    ["deep", "learning", "uses", "neural", "networks"],
    ["neural", "networks", "are", "powerful"],
    ["i", "love", "artificial", "intelligence"]
]

from gensim.models import Word2Vec

model = Word2Vec(sentences, vector_size=10, window=5, min_count=1, workers=4, sg = 1 , epochs=10)


vector = model.wv["machine"]

print(vector)
print(len(vector))

similar_words = model.wv.most_similar("learning")
print(similar_words)
