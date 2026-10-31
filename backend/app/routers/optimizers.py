import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense ,Dropout
from tensorflow.keras.optimizers import SGD   ## you can use any  optimizers

x,y = make_moons(n_samples=1000 , noise=0.2 , random_state= 30)

X_train, X_test, y_train, y_test = train_test_split(
x, y, test_size=0.2, random_state=30
)

scalar = StandardScaler()

X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)


model = Sequential([
    Dense(64, activation="relu", input_shape=(2,)),
    Dropout(0.5),
    Dense(32, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

momentum_optimizer = SGD(
    learning_rate=0.001,
    momentum=0.9
)

model.compile(
    optimizer=momentum_optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train ,
    y_train ,
    epochs =30 ,
    batch_size = 40 ,
    validation_split = 0.2 ,
    verbose = 1
)

model.summary()

test_loss, test_accuracy = model.evaluate(
    X_test, y_test, verbose=0
)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)