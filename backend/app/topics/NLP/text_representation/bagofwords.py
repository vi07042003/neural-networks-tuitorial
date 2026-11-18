from sklearn.feature_extraction.text import CountVectorizer

sentences = [
    "This is the first sentence.",
    "This is the second sentence.",
    "And this is the third sentence."
]

vectorizer = CountVectorizer()
bag_of_words = vectorizer.fit_transform(sentences)
print(vectorizer.get_feature_names_out())
print(bag_of_words.toarray())

