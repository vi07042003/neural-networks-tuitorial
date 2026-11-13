import tensorflow as tf
from  tensorflow.keras import layers, models

##load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

## nomralize the data
x_train = x_train / 255.0
x_test = x_test / 255.0

## add a channels dimension
x_train = x_train[..., None]
x_test = x_test[..., None]

## cnn model
model = models.Sequential([layers.Input(shape=(28, 28, 1)),
                           layers.Conv2D(32, kernel_size=(3, 3), activation='relu'), ## convolution layer with 32 filters and kernel size of 3x3
                           layers.MaxPooling2D(pool_size=(2, 2)), ## max pooling layer with pool size of 2x2
                           layers.Conv2D(64, kernel_size=(3, 3), activation='relu'), ## convolution layer with 64 filters and kernel size of 3x3
                           layers.MaxPooling2D(pool_size=(2, 2)), ## max pooling layer with pool size of 2x2
                            layers.Flatten(), ## flatten the output of the previous layer
                            layers.Dense(128, activation='relu'), ## dense layer with 128 units and relu activation
                            layers.Dense(10, activation='softmax') ## output layer with 10 units and softmax activation
                           ])

### compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])   

### train the model
model.fit(x_train, y_train, 
        batch_size = 32,
        epochs=5,
        validation_data=(x_test, y_test))

### evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")