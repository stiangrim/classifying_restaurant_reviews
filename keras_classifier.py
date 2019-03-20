import pickle
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing.sequence import pad_sequences

# Loading the data from file
data = pickle.load(open("data/keras-data.pickle", "rb"))

# Setting variables
x_train = data['x_train']
y_train = data['y_train']
x_test = data['x_test']
y_test = data['y_test']
vocab_size = data['vocab_size']
max_length = data['max_length']

# Padding sequences
x_train = pad_sequences(sequences=x_train, maxlen=max_length)
x_test = pad_sequences(sequences=x_test, maxlen=max_length)

# Creating the model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=50))
model.add(LSTM(units=50, dropout=0.0, return_sequences=False))
model.add(Dense(units=1, activation='relu'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=2, batch_size=128, verbose=1)
model.evaluate(x_test, y_test, verbose=1)