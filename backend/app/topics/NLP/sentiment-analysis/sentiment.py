reviews = [
    "This product is amazing",
    "I love this phone",
    "The quality is excellent",
    "This is a wonderful experience",
    "I hate this product",
    "The phone is terrible",
    "Very bad quality",
    "I am disappointed",
    "The product arrived today",
    "The phone has a large screen",
    "The package was delivered yesterday",
    "It has three buttons"
]

labels = [
    "positive",
    "positive",
    "positive",
    "positive",
    "negative",
    "negative",
    "negative",
    "negative",
    "neutral",
    "neutral",
    "neutral",
    "neutral"
]

## split the dataset

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(reviews,
labels, test_size=0.2,
random_state=42,stratify=labels)



## convert text into numbers using tf_idf vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)


##train model using logistic regression model

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train_vector , y_train)

## make predictions

predictions = model.predict(X_test_vector)


print(predictions)


new_review = ["i hate this mobile "]

new_review_vector = vectorizer.transform(new_review)

result = model.predict(new_review_vector)

print("Sentiment:", result[0])


## evaluate the model 

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


print("accuracy", accuracy_score(y_test, predictions))

print(classification_report(y_test,predictions,zero_division=0))

print(confusion_matrix(y_test,predictions))