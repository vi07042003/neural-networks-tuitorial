import nltk

from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

words = ["cars", "running", "better"]

for word in words:
    print(word, ":", lemmatizer.lemmatize(word))