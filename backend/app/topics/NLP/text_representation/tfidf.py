from sklearn.feature_extraction.text import TfidfVectorizer

sentences = [
    "This is the first sentence.",
    "This is the second sentence.",
    "And this is the third sentence."
]

vectorizer = TfidfVectorizer()

results = vectorizer.fit_transform(sentences)

print("Vocabulary:")
print(vectorizer.get_feature_names_out())

print("\nTF-IDF values:")
print(results.toarray())