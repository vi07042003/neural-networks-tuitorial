## messages and  labels for traning the model
messages = [
    "Congratulations you won a lottery",
    "Win a free iPhone now",
    "Claim your free prize",
    "Hey how are you",
    "Let's meet tomorrow",
    "Are you coming to the office",
    "You won free cash",
    "I will call you later",
    "Get a free offer now",
    "See you tomorrow"
]

labels = [
    "spam",
    "spam",
    "spam",
    "not_spam",
    "not_spam",
    "not_spam",
    "spam",
    "not_spam",
    "spam",
    "not_spam"
]


## split the dataset

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(messages,
labels, test_size=0.2,
random_state=42,stratify=labels)


## convert text into numbers using tf_idf vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)


##train model usng navie bayes model

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

model.fit(X_train_vector , y_train)

## make predictions

predictions = model.predict(X_test_vector)


print(predictions)


## evaluate the model 

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


print("accuracy", accuracy_score(y_test, predictions))

print(classification_report(y_test,predictions,zero_division=0))

print(confusion_matrix(y_test,predictions))


