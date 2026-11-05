import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten , Dense 
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt 


(x_train , y_train) ,(x_test , y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train/225.0
x_test = x_test/225.0


model = Sequential([Flatten(input_shape = (28,28)),
                    Dense (128 ,activation="relu", kernel_regularizer=regularizers.l2(0.2)),
                    Dense (10 ,activation="softmax")
                    ])

model.compile(
    optimizer="adam",
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    x_train ,
    y_train ,
    epochs =20 ,
    batch_size = 32 ,
    validation_split = 0.2 ,
    callbacks = [early_stopping],
    verbose = 1
)

test_loss, test_accuracy = model.evaluate(x_test, y_test)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")



plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.savefig("training_loss.png")
plt.show()