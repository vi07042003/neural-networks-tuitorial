from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

words = ["playing", "played", "plays", "player"]

for word in words:
    print(word, ":", stemmer.stem(word))