import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input ,Flatten, Dense ,Dropout
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV


##load dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


## defining the model architecture create nn
def create_model(learning_rate=0.001, dropout_rate=0.5):
    model = Sequential([
    Input(shape=(28, 28)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


##defining the hyperparameter grid

model  = KerasClassifier(model=create_model, verbose=0)

param_grid = {
    'model__learning_rate': [0.001, 0.01, 0.1],
    'model__dropout_rate': [0.3, 0.5, 0.7], 
    'epochs': [5, 10] ,
    'batch_size': [32, 64]
}

## using grid serach 
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3 )
grid_search.fit(x_train, y_train)

## print results
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

#3 evualuate the best model on the test set
best_model = grid_search.best_estimator_.model
test_loss, test_accuracy = best_model.evaluate(x_test, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")