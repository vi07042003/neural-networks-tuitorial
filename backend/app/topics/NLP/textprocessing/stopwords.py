from nltk import word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
nltk.download('punkt_tab')

text = "This is a sample sentence, showing off the stop words filtration."
stop_words = set(stopwords.words('english'))
tokens = word_tokenize(text)
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
print(filtered_tokens)