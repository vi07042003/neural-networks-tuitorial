import tensorflow as tf
import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense ,SimpleRNN


#defininf the text and creating a mapping of characters to indices and vice versa
text = "i am vinit"
chars = sorted(list(set(text)))
char_to_index = {char: i for i, char in enumerate(chars)}
index_to_char = {i: char for i, char in enumerate(chars)}


#creating sequences and labels
seq_len = 3
sequences = []
labels = []
for i in range(len(text) - seq_len):
    seq = text[i:i + seq_len]
    label = text[i + seq_len]
    sequences.append([char_to_index[char] for char in seq])
    labels.append(char_to_index[label]) 

x = np.array(sequences)
y = np.array(labels)

#converting sequences and labels to one-hot encoding
x_one_hot = tf.one_hot(x, len(chars))
y_one_hot = tf.one_hot(y, len(chars))


##build the rnn model 
model = Sequential()
model.add(SimpleRNN(32, input_shape=(seq_len, len(chars)), activation='relu'))
model.add(Dense(len(chars), activation='softmax'))

##compiling the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_one_hot, y_one_hot, epochs=100)

## generating text using the trained model

start_seq = "am"
generated_text = start_seq

for i in range(50):
    x = np.array([[char_to_index[char] for char in generated_text[-seq_len:]]])
    x_one_hot = tf.one_hot(x, len(chars))
    prediction = model.predict(x_one_hot)
    next_index = np.argmax(prediction)
    next_char = index_to_char[next_index]
    generated_text += next_char

print("Generated Text:")
print(generated_text)