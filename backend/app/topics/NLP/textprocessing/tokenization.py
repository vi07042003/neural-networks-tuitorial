from nltk import word_tokenize
import nltk

nltk.download('punkt_tab')

text = "Hello, how are you doing today?"

tokens = word_tokenize(text)
print(tokens)