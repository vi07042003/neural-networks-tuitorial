from sklearn.feature_extraction.text import CountVectorizer

sentences = [
    "I love machine learning",
    "I love deep learning"
]

# Fix: Use ngram_range with a tuple instead of a range object
vectorizer = CountVectorizer(ngram_range=(1, 2))
bag_of_words = vectorizer.fit_transform(sentences)

print(vectorizer.get_feature_names_out())
print(bag_of_words.toarray())