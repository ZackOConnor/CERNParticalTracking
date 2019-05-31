from keras.models import Sequential
from keras.models import *
from keras.layers import *
from keras.layers.embeddings import Embedding


def make_model():
    model = Sequential()
    model.add(Embedding(50000, 100, input_length=11))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))
    #Builds the LSTM, with 2 layers and reasonable drop out

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    # Defines loss function and prediction metric

    return model