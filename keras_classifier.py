import pickle
from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM
from keras.preprocessing.sequence import pad_sequences

# Set variables
data = pickle.load(open("TDT4171-Ex5/keras-data.pickle", "rb"))
x_train = data['x_train']
y_train = data['y_train']
x_test = data['x_test']
y_test = data['y_test']
vocab_size = data['vocab_size']
max_length = data['max_length']

# Do padding
max_length = 200
x_train = pad_sequences(sequences=x_train, maxlen=max_length)
x_test = pad_sequences(sequences=x_test, maxlen=max_length)

# Creating the model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=50, input_length=max_length))
model.add(LSTM(units=50, dropout=0.2))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Fit model
model.fit(x_train, y_train, epochs=10, batch_size=128, verbose=1)

# Evaluate model
model.evaluate(x_test, y_test, verbose=1)